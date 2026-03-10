/*
 * gpu/gpu_hashtable.cu
 *
 * GPU-side hash table lookup:
 *   - Upload target address hash table (ht_slots) to GPU VRAM and keep resident
 *   - Execute Hash160 lookup directly on GPU
 *   - Transfer only hit private key info back to CPU
 *   - Fall back to CPU-side lookup mode if hash table exceeds VRAM capacity
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4
 */

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gpu_interface.h"
#include "../hash_utils.h"
#include "../keylog.h"


/*
 * Device-side hash table slot (layout consistent with CPU-side struct ht_slot)
 * fp    : first 4-byte fingerprint (big-endian)
 * h160  : full 20-byte Hash160
 */
struct gpu_ht_slot {
    uint32_t fp;
    uint8_t  h160[20];
};

static struct gpu_ht_slot *d_ht_slots = NULL;
static uint32_t d_ht_mask_val = 0;
static int g_use_gpu_ht = 0; /* 1=GPU-side lookup, 0=CPU-side lookup */

/* Hit result device buffers */
static gpu_hit_result_t *d_hit_results = NULL;
static int *d_hit_count = NULL; /* atomic counter */

/*
 * Kernel: batch Hash160 lookup
 * Optimization: each thread handles one (chain, step) pair, matching the parallelism
 * granularity of kernel_hash160.
 * Total threads = num_chains * steps; each thread looks up compressed/uncompressed Hash160
 * for one public key.
 */
__global__ void kernel_hashtable_lookup(
    const uint8_t          * __restrict__ hash160_comp,
    const uint8_t          * __restrict__ hash160_uncomp,
    const int              * __restrict__ valid,
    const uint8_t          * __restrict__ base_privkeys,
    const struct gpu_ht_slot * __restrict__ ht_slots,
    uint32_t ht_mask,
    gpu_hit_result_t       * __restrict__ hit_results,
    int                    * __restrict__ hit_count,
    int num_chains,
    int steps)
{
    /* Each thread handles one (chain, step) pair */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chains * steps;
    if (tid >= total)
        return;

    int chain = tid / steps;
    int step  = tid % steps;

    /* Skip invalid steps */
    if (step >= valid[chain])
        return;

    int idx = chain * steps + step;
    const uint8_t *base_priv = base_privkeys + chain * 32;
    const uint8_t *h160c = hash160_comp + idx * 20;
    const uint8_t *h160uc = hash160_uncomp + idx * 20;

    /* Look up compressed pubkey Hash160 */
    uint32_t fp_c = ((uint32_t)h160c[0] << 24) | ((uint32_t)h160c[1] << 16) |
                    ((uint32_t)h160c[2] <<  8) | ((uint32_t)h160c[3]);
    uint32_t slot_c = fp_c & ht_mask;
    for (;;) {
        const struct gpu_ht_slot *s = &ht_slots[slot_c];
        if (s->fp == 0)
            break;  /* empty slot, no hit */
        if (s->fp == fp_c) {
            /* Fingerprint match, verify full 20 bytes */
            int match = 1;
            for (int k = 0; k < 20; k++) {
                if (s->h160[k] != h160c[k]) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                /* Hit! Write to result buffer */
                int pos = atomicAdd(hit_count, 1);
                if (pos < GPU_MAX_HITS) {
                    gpu_hit_result_t *r = &hit_results[pos];
                    /* Rebuild private key: base_priv + step (simplified: store base_priv and step offset) */
                    for (int k = 0; k < 32; k++)
                        r->privkey[k] = base_priv[k];
                    /* Encode step into last 4 bytes of privkey (convention: caller rebuilds) */
                    r->privkey[28] = (uint8_t)(step >> 24);
                    r->privkey[29] = (uint8_t)(step >> 16);
                    r->privkey[30] = (uint8_t)(step >> 8);
                    r->privkey[31] = (uint8_t)(step);
                    for (int k = 0; k < 20; k++)
                        r->hash160[k] = h160c[k];
                    r->is_compressed = 1;
                }
                break;
            }
        }
        slot_c = (slot_c + 1) & ht_mask;  /* linear probing */
    }

    /* Look up uncompressed pubkey Hash160 */
    uint32_t fp_uc = ((uint32_t)h160uc[0] << 24) | ((uint32_t)h160uc[1] << 16) |
                        ((uint32_t)h160uc[2] <<  8) | ((uint32_t)h160uc[3]);
    uint32_t slot_uc = fp_uc & ht_mask;
    for (;;) {
        const struct gpu_ht_slot *s = &ht_slots[slot_uc];
        if (s->fp == 0)
            break;
        if (s->fp == fp_uc) {
            int match = 1;
            for (int k = 0; k < 20; k++) {
                if (s->h160[k] != h160uc[k]) {
                    match = 0;
                    break;
                }
            }
            if (match) {
                int pos = atomicAdd(hit_count, 1);
                if (pos < GPU_MAX_HITS) {
                    gpu_hit_result_t *r = &hit_results[pos];
                    for (int k = 0; k < 32; k++)
                        r->privkey[k] = base_priv[k];
                    r->privkey[28] = (uint8_t)(step >> 24);
                    r->privkey[29] = (uint8_t)(step >> 16);
                    r->privkey[30] = (uint8_t)(step >> 8);
                    r->privkey[31] = (uint8_t)(step);
                    for (int k = 0; k < 20; k++)
                        r->hash160[k] = h160uc[k];
                    r->is_compressed = 0;
                }
                break;
            }
        }
        slot_uc = (slot_uc + 1) & ht_mask;  /* linear probing */
    }
}


/*
 * ht_slots_cpu : CPU-side struct ht_slot array (provided by hash_utils.h)
 * ht_capacity  : slot count (power of 2)
 * free_mem     : GPU available VRAM (bytes), used to determine if hash table fits
 * Return value: 1=GPU-side lookup, 0=fall back to CPU-side lookup, -1=fatal error
 */
int gpu_hashtable_upload(const struct ht_slot *ht_slots_cpu,
    uint32_t ht_capacity, size_t free_mem)
{
    size_t ht_size = (size_t)ht_capacity * sizeof(struct gpu_ht_slot);

    /* Check VRAM sufficiency (reserve 64MB for other buffers) */
    if (ht_size + 64ULL * 1024 * 1024 > free_mem) {
        char ht_str[32], free_str[32];
        snprintf(ht_str, sizeof(ht_str),  "%.0f MB", (double)ht_size / (1024.0 * 1024.0));
        snprintf(free_str, sizeof(free_str), "%.0f MB", (double)free_mem / (1024.0 * 1024.0));
        keylog_warn("[GPU] Hash table requires %s, available VRAM %s, falling back to CPU-side lookup",
                    ht_str, free_str);
        g_use_gpu_ht = 0;
        return 0;
    }

    /* Allocate device-side hash table */
    if (cudaMalloc(&d_ht_slots, ht_size) != cudaSuccess) {
        keylog_warn("[GPU] Hash table VRAM allocation failed, falling back to CPU-side lookup");
        g_use_gpu_ht = 0;
        return 0;
    }

    /* Upload hash table (once, remains resident throughout search) */
    if (cudaMemcpy(d_ht_slots, ht_slots_cpu, ht_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        keylog_error("[GPU] Hash table upload failed");
        cudaFree(d_ht_slots);
        d_ht_slots = NULL;
        return -1;
    }

    d_ht_mask_val = ht_capacity - 1;

    /* Allocate hit result buffer */
    if (cudaMalloc(&d_hit_results, GPU_MAX_HITS * sizeof(gpu_hit_result_t)) != cudaSuccess) {
        keylog_error("[GPU] Hit result buffer allocation failed");
        return -1;
    }
    if (cudaMalloc(&d_hit_count, sizeof(int)) != cudaSuccess) {
        keylog_error("[GPU] Hit counter allocation failed");
        return -1;
    }

    char ht_str[32];
    snprintf(ht_str, sizeof(ht_str), "%.0f MB", (double)ht_size / (1024.0 * 1024.0));
    keylog_info("[GPU] Hash table uploaded to GPU VRAM (%s, %u slots), GPU-side lookup enabled",
                ht_str, ht_capacity);
    g_use_gpu_ht = 1;
    return 1;
}

/*
 * Free GPU-side hash table resources
 */
void gpu_hashtable_free(void)
{
    if (d_ht_slots) {
        cudaFree(d_ht_slots);
        d_ht_slots = NULL;
    }
    if (d_hit_results) {
        cudaFree(d_hit_results);
        d_hit_results = NULL;
    }
    if (d_hit_count) {
        cudaFree(d_hit_count);
        d_hit_count = NULL;
    }
}

/*
 * Execute GPU-side hash table lookup kernel
 * Returns hit count (written to h_hits), -1 on error
 *
 * Note: privkey[28..31] in hit results stores the step offset,
 *       caller (gpu_search.cu) needs to rebuild the full private key
 */
int gpu_hashtable_run(
    const uint8_t *d_hash160_comp,
    const uint8_t *d_hash160_uncomp,
    const int     *d_valid,
    const uint8_t *d_base_privkeys,
    int num_chains, int steps,
    gpu_hit_result_t *h_hits, int *h_hit_count,
    cudaStream_t stream)
{
    if (!g_use_gpu_ht)
        return 0;  /* fallback mode, do not execute */

    /* Reset hit counter */
    cudaMemsetAsync(d_hit_count, 0, sizeof(int), stream);

    /* Each thread handles one (chain, step) pair; total threads = num_chains * steps */
    int block_size = 256;
    int total_threads = num_chains * steps;
    int grid_size = (total_threads + block_size - 1) / block_size;

    kernel_hashtable_lookup<<<grid_size, block_size, 0, stream>>>(
        d_hash160_comp, d_hash160_uncomp,
        d_valid, d_base_privkeys,
        d_ht_slots, d_ht_mask_val,
        d_hit_results, d_hit_count,
        num_chains, steps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        keylog_error("[GPU] kernel_hashtable_lookup execution failed: %s",
                     cudaGetErrorString(err));
        return -1;
    }

    /* Wait for all operations on the stream to complete */
    cudaStreamSynchronize(stream);

    /* Transfer hit count and results back (only when there are hits) */
    int hit_count = 0;
    cudaMemcpy(&hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    *h_hit_count = hit_count;

    if (hit_count > 0) {
        int actual = (hit_count > GPU_MAX_HITS) ? GPU_MAX_HITS : hit_count;
        cudaMemcpy(h_hits, d_hit_results, actual * sizeof(gpu_hit_result_t), cudaMemcpyDeviceToHost);
    }

    return hit_count;
}

/* Return whether GPU-side lookup is in use */
int gpu_hashtable_is_gpu_mode(void)
{
    return g_use_gpu_ht;
}
