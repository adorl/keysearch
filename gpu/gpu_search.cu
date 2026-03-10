/*
 * gpu/gpu_search.cu
 *
 * GPU search main loop (gpu_search() entry point):
 *   - Calls gpu_detect.cu for device initialization
 *   - Allocates GPU work buffers
 *   - Double-buffering pipeline main loop:
 *       while GPU processes buf[A] (pubkey/hash160/lookup) on stream[A],
 *       CPU generates random private keys into buf[B] and transfers them
 *       asynchronously via cudaMemcpyAsync; two CUDA streams alternate to
 *       eliminate CPU/PCIe serial stalls.
 *   - On hit, calls keylog_info to record private key and address
 *   - Prints speed info every PROGRESS_INTERVAL iterations
 *
 * Requirements: 3.2, 8.1, 8.2, 8.3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#include "gpu_interface.h"
#include "../keylog.h"
#include "../hash_utils.h"
#include "../rand_key.h"

/* External Function Declarations (from other .cu files) */

/* gpu_detect.cu */
extern int gpu_detect_and_select(uint32_t ht_capacity, int *out_batch);

/* gpu_secp256k1.cu */
extern int gpu_secp256k1_alloc(int num_chains);
extern void gpu_secp256k1_free(void);
extern int gpu_secp256k1_run(const uint8_t *h_base_privkeys, cudaStream_t stream);
extern const uint8_t *gpu_secp256k1_get_aff_x(void);
extern const uint8_t *gpu_secp256k1_get_aff_y(void);
extern const int *gpu_secp256k1_get_valid(void);
extern int gpu_secp256k1_get_num_chains(void);
extern const uint8_t *gpu_secp256k1_get_base_privkeys(void);

/* gpu_hash160.cu */
extern int gpu_hash160_alloc(int num_chains, int steps);
extern void gpu_hash160_free(void);
extern int gpu_hash160_run(const uint8_t *d_aff_x, const uint8_t *d_aff_y,
    const int *d_valid, cudaStream_t stream);
extern const uint8_t *gpu_hash160_get_comp(void);
extern const uint8_t *gpu_hash160_get_uncomp(void);
extern int gpu_hash160_get_num_chains(void);
extern int gpu_hash160_get_steps(void);

/* gpu_hashtable.cu */
extern int gpu_hashtable_upload(const struct ht_slot *ht_slots_cpu,
    uint32_t ht_capacity, size_t free_mem);
extern void gpu_hashtable_free(void);
extern int gpu_hashtable_run(const uint8_t *d_hash160_comp,
    const uint8_t *d_hash160_uncomp, const int *d_valid,
    const uint8_t *d_base_privkeys, int num_chains, int steps,
    gpu_hit_result_t *h_hits, int *h_hit_count, cudaStream_t stream);
extern int gpu_hashtable_is_gpu_mode(void);

static int g_initialized = 0;
static int g_batch_size  = 0;

/* ---- Double-buffer structure ---- */
#define NUM_BUFS 2

/*
 * Each buffer contains:
 *   h_base_privkeys : CPU-side pinned memory for async DMA transfer
 */
typedef struct {
    uint8_t *h_base_privkeys;   /* CPU pinned memory */
} gpu_buf_t;

static gpu_buf_t g_bufs[NUM_BUFS];
static cudaStream_t g_streams[NUM_BUFS];

/* CPU-side Hash160 buffer (used when falling back to CPU-side lookup) */
static uint8_t *h_hash160_comp   = NULL;
static uint8_t *h_hash160_uncomp = NULL;

/* CPU-side valid step count buffer (used when falling back to CPU-side lookup) */
static int *h_valid = NULL;

/* Hit result buffer */
static gpu_hit_result_t h_hits[GPU_MAX_HITS];

/* Helper Functions */

/*
 * Rebuild hit private key: base_privkey + step times +1
 * Note: gpu_hashtable.cu encodes step into privkey[28..31]
 */
static void rebuild_privkey(const uint8_t *base_priv, uint8_t *out_priv)
{
    /* Extract step (stored in privkey[28..31]) */
    int step = ((int)base_priv[28] << 24) | ((int)base_priv[29] << 16) |
               ((int)base_priv[30] <<  8) | ((int)base_priv[31]);

    memcpy(out_priv, base_priv, 32);

    /* Perform +step operation on 256-bit integer */
    uint64_t carry = (uint64_t)step;
    for (int i = 31; i >= 0 && carry > 0; i--) {
        uint64_t sum = (uint64_t)out_priv[i] + carry;
        out_priv[i] = (uint8_t)(sum & 0xFF);
        carry = sum >> 8;
    }
}

/*
 * Handle hit result: rebuild private key, compute address, log
 */
static void handle_hit(const gpu_hit_result_t *hit, uint64_t total_count)
{
    uint8_t privkey[32];
    rebuild_privkey(hit->privkey, privkey);

    char privkey_hex[65];
    bytes_to_hex(privkey, 32, privkey_hex);

    char addr_compressed[36];
    char addr_uncompressed[36];

    keylog_info("[GPU] Match found! Total attempts: %lu", total_count);
    keylog_info("[GPU] Private key (hex): %s", privkey_hex);
    keylog_info("[GPU] Hit type: %s", hit->is_compressed ? "compressed address" : "uncompressed address");

    if (privkey_to_address(privkey, addr_compressed, addr_uncompressed) == 0) {
        keylog_info("[GPU] Compressed address: %s",   addr_compressed);
        keylog_info("[GPU] Uncompressed address: %s", addr_uncompressed);
    }
}

/*
 * CPU-side lookup fallback: transfer Hash160 from GPU then lookup on CPU
 * cur_buf : index of the buffer currently being processed, used to retrieve
 *            the corresponding CPU-side private keys
 */
static int cpu_fallback_lookup(int num_chains, int steps, int cur_buf,
    uint64_t *total_count, uint64_t *count_last, struct timespec *ts_last)
{
    /* Transfer Hash160 results back (num_chains * steps * 20 * 2 bytes) */
    size_t sz = (size_t)num_chains * steps * 20;
    cudaMemcpy(h_hash160_comp, gpu_hash160_get_comp(), sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hash160_uncomp, gpu_hash160_get_uncomp(), sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid, gpu_secp256k1_get_valid(), (size_t)num_chains * sizeof(int), cudaMemcpyDeviceToHost);

    int found = 0;
    for (int chain = 0; chain < num_chains; chain++) {
        int valid_steps = h_valid[chain];
        for (int step = 0; step < valid_steps; step++) {
            int idx = chain * steps + step;
            const uint8_t *h160c = h_hash160_comp + idx * 20;
            const uint8_t *h160uc = h_hash160_uncomp + idx * 20;

            (*total_count)++;

            if (ht_contains(h160c) || ht_contains(h160uc)) {
                /* Rebuild private key */
                gpu_hit_result_t hit;
                /* Use private keys from the currently active buffer */
                memcpy(hit.privkey, g_bufs[cur_buf].h_base_privkeys + chain * 32, 32);
                /* Encode step into privkey[28..31] */
                hit.privkey[28] = (uint8_t)(step >> 24);
                hit.privkey[29] = (uint8_t)(step >> 16);
                hit.privkey[30] = (uint8_t)(step >>  8);
                hit.privkey[31] = (uint8_t)(step);
                if (ht_contains(h160c)) {
                    memcpy(hit.hash160, h160c, 20);
                    hit.is_compressed = 1;
                }
                else {
                    memcpy(hit.hash160, h160uc, 20);
                    hit.is_compressed = 0;
                }
                handle_hit(&hit, *total_count);
                found++;
            }
        }
    }
    return found;
}

/*
 * gpu_init: initialize GPU, upload hash table
 */
int gpu_init(const void *ht_slots_ptr, uint32_t ht_capacity)
{
    int batch_size = 0;

    /* Detect and select GPU device */
    if (gpu_detect_and_select(ht_capacity, &batch_size) != 0)
        return -1;

    g_batch_size = batch_size;

    /* Allocate secp256k1 work buffers */
    if (gpu_secp256k1_alloc(batch_size) != 0)
        return -1;

    /* Allocate Hash160 buffers */
    if (gpu_hash160_alloc(batch_size, GPU_CHAIN_STEPS) != 0)
        return -1;

    /* Upload hash table to GPU VRAM */
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    int ht_ret = gpu_hashtable_upload(
        (const struct ht_slot *)ht_slots_ptr, ht_capacity, free_mem);
    if (ht_ret < 0)
        return -1;

    /* ---- Allocate double buffers and CUDA streams ---- */
    for (int i = 0; i < NUM_BUFS; i++) {
        /* CPU-side pinned memory (supports async DMA) */
        if (cudaMallocHost(&g_bufs[i].h_base_privkeys, (size_t)batch_size * 32) != cudaSuccess) {
            keylog_error("[GPU] CPU pinned memory allocation failed (buf %d)", i);
            return -1;
        }
        /* Create CUDA stream */
        if (cudaStreamCreate(&g_streams[i]) != cudaSuccess) {
            keylog_error("[GPU] CUDA stream creation failed (stream %d)", i);
            return -1;
        }
    }

    /* CPU-side lookup fallback buffers */
    if (!gpu_hashtable_is_gpu_mode()) {
        size_t h160_sz = (size_t)batch_size * GPU_CHAIN_STEPS * 20;
        h_hash160_comp = (uint8_t *)malloc(h160_sz);
        h_hash160_uncomp = (uint8_t *)malloc(h160_sz);
        h_valid = (int *)malloc((size_t)batch_size * sizeof(int));
        if (!h_hash160_comp || !h_hash160_uncomp || !h_valid) {
            keylog_error("[GPU] CPU-side Hash160 buffer allocation failed");
            return -1;
        }
    }

    const gpu_device_info_t *info = gpu_get_device_info();
    int grid_size = (batch_size + 255) / 256;
    keylog_info("[GPU] Initialization complete: device=%s | batch=%d | grid=%dx256 threads | steps=%d | double-buffering enabled",
                info->name, batch_size, grid_size, GPU_CHAIN_STEPS);

    g_initialized = 1;
    return 0;
}

/*
 * Generate a batch of random private keys into the specified buffer
 */
static int fill_random_privkeys(gpu_buf_t *buf, int num_chains, rand_key_context *rand_ctx)
{
    for (int i = 0; i < num_chains; i++) {
        uint8_t *priv = buf->h_base_privkeys + i * 32;
        int ok = 0;
        while (!ok) {
            if (gen_random_key(priv, rand_ctx) != 0) {
                keylog_error("[GPU] Failed to read random number");
                return -1;
            }
            /* Simple check: non-zero */
            int all_zero = 1;
            for (int k = 0; k < 32; k++) {
                if (priv[k] != 0) {
                    all_zero = 0;
                    break;
                }
            }
            ok = !all_zero;
        }
    }
    return 0;
}

/*
 * gpu_search: GPU search main loop (double-buffering pipeline)
 *
 * Pipeline timing (steady state):
 *   Round N  : GPU executes pubkey+hash160+lookup on stream[N%2] (processing buf[N%2])
 *   Round N  : CPU concurrently generates random private keys into buf[(N+1)%2]
 *   Round N+1: GPU executes on stream[(N+1)%2] (processing buf[(N+1)%2])
 *              CPU concurrently generates the next batch of private keys into buf[N%2]
 */
int gpu_search(void)
{
    if (!g_initialized) {
        keylog_error("[GPU] gpu_search called before initialization");
        return -1;
    }

    rand_key_context rand_ctx;
    if (rand_ctx_init(&rand_ctx) != 0) {
        keylog_error("[GPU] Failed to open /dev/urandom");
        return -1;
    }

    uint64_t total_count = 0;
    uint64_t count_last  = 0;
    struct timespec ts_last, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_last);

    int progress_counter = PROGRESS_INTERVAL;
    int num_chains = g_batch_size;
    int steps = GPU_CHAIN_STEPS;

    keylog_info("[GPU] Starting search (double-buffering pipeline), batch=%d, steps=%d, total %d pubkeys/batch",
                num_chains, steps, num_chains * steps);

    /* ---- Pre-fill buf[0] to kick off the first round ---- */
    if (fill_random_privkeys(&g_bufs[0], num_chains, &rand_ctx) != 0)
        return -1;

    int cur = 0;  /* index of the buffer currently being processed by GPU */

    while (1) {
        int nxt = 1 - cur;  /* index of the next buffer */

        /* ---- Submit three kernels on stream[cur] ---- */
        /* gpu_secp256k1_run internally transfers private keys via cudaMemcpyAsync and launches kernel */
        if (gpu_secp256k1_run(g_bufs[cur].h_base_privkeys, g_streams[cur]) != 0)
            return -1;

        if (gpu_hash160_run(gpu_secp256k1_get_aff_x(), gpu_secp256k1_get_aff_y(),
                            gpu_secp256k1_get_valid(), g_streams[cur]) != 0)
            return -1;

        /* ---- CPU concurrently generates the next batch of random private keys (overlaps GPU) ---- */
        if (fill_random_privkeys(&g_bufs[nxt], num_chains, &rand_ctx) != 0)
            return -1;

        /* ---- Wait for hash160 kernel on stream[cur] to finish, then run lookup ---- */
        if (gpu_hashtable_is_gpu_mode()) {
            int hit_count = 0;
            int ret = gpu_hashtable_run(gpu_hash160_get_comp(), gpu_hash160_get_uncomp(),
                gpu_secp256k1_get_valid(), gpu_secp256k1_get_base_privkeys(),  /* internal d_base_privkeys from secp256k1 module */
                num_chains, steps, h_hits, &hit_count, g_streams[cur]);
            if (ret < 0)
                return -1;

            /* gpu_hashtable_run already called cudaStreamSynchronize(stream[cur]) internally */
            total_count += (uint64_t)num_chains * steps;

            /* Process hit results (private keys come from buf[cur]) */
            int actual_hits = (hit_count > GPU_MAX_HITS) ? GPU_MAX_HITS : hit_count;
            for (int i = 0; i < actual_hits; i++) {
                handle_hit(&h_hits[i], total_count);
            }
        } else {
            /* CPU-side lookup fallback */
            cudaStreamSynchronize(g_streams[cur]);
            cpu_fallback_lookup(num_chains, steps, cur, &total_count, &count_last, &ts_last);
        }

        /* ---- Performance monitoring ---- */
        progress_counter -= num_chains * steps;
        if (progress_counter <= 0) {
            progress_counter = PROGRESS_INTERVAL;
            clock_gettime(CLOCK_MONOTONIC, &ts_now);
            double elapsed = (ts_now.tv_sec  - ts_last.tv_sec) +
                             (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
            double kps = (elapsed > 0) ? (double)(total_count - count_last) / elapsed : 0.0;
            keylog_info("[GPU] Attempts: %lu Speed: %.0f keys/s", total_count, kps);
            ts_last = ts_now;
            count_last = total_count;
        }

        /* Switch to next buffer */
        cur = nxt;
    }

    return 0;
}

/*
 * gpu_cleanup: free all GPU resources
 */
void gpu_cleanup(void)
{
    gpu_secp256k1_free();
    gpu_hash160_free();
    gpu_hashtable_free();

    for (int i = 0; i < NUM_BUFS; i++) {
        if (g_bufs[i].h_base_privkeys) {
            cudaFreeHost(g_bufs[i].h_base_privkeys);
            g_bufs[i].h_base_privkeys = NULL;
        }
        if (g_streams[i]) {
            cudaStreamDestroy(g_streams[i]);
            g_streams[i] = NULL;
        }
    }

    free(h_hash160_comp);
    h_hash160_comp = NULL;
    free(h_hash160_uncomp);
    h_hash160_uncomp = NULL;
    free(h_valid);
    h_valid = NULL;

    g_initialized = 0;
    keylog_info("[GPU] Resources freed");
}
