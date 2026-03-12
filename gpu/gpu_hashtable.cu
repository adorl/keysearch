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
#include "gpu_hash160_impl.cuh"
#include "../hash_utils.h"
#include "../keylog.h"


/*
 * Device-side hash table slot (layout consistent with CPU-side struct ht_slot)
 * fp    : first 4-byte fingerprint (big-endian)
 * h160  : full 20-byte Hash160
 * _pad  : padding to 32 bytes for aligned global memory access
 *
 * H2 optimization: align to 32 bytes so each slot fits exactly in one 32-byte
 * cache line, eliminating cross-line reads and enabling coalesced warp access.
 */
struct __align__(32) gpu_ht_slot {
    uint32_t fp;
    uint8_t  h160[20];
    uint8_t  _pad[8];   /* padding: 4 + 20 + 8 = 32 bytes */
};

static struct gpu_ht_slot *d_ht_slots = NULL;
static uint32_t d_ht_mask_val = 0;
static int g_use_gpu_ht = 0; /* 1=GPU-side lookup, 0=CPU-side lookup */

/* Hit result device buffers */
static gpu_hit_result_t *d_hit_results = NULL;
static int *d_hit_count = NULL; /* atomic counter */

/*
 * Kernel: batch Hash160 lookup
 *
 * H2 optimization: 32-byte aligned gpu_ht_slot
 *   Each slot is exactly 32 bytes (one cache line), so every slot read is a
 *   single aligned 32-byte transaction. __ldg() routes reads through the
 *   read-only texture cache, reducing L2 pressure for the hash table.
 *
 * H3 optimization: steps must be a power of 2 (enforced by caller).
 *   chain = tid >> steps_log2  (bit-shift replaces integer division)
 *   step  = tid &  steps_mask  (bit-AND replaces modulo)
 *
 * Total threads = num_chains * steps (unchanged from original layout).
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
    int steps,
    int steps_log2)   /* H3: log2(steps), used for bit-shift instead of division */
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chains * steps;
    if (tid >= total)
        return;

    /* H3: replace division/modulo with bit operations (steps is power of 2) */
    int steps_mask = steps - 1;
    int chain = tid >> steps_log2;   /* tid / steps */
    int step  = tid &  steps_mask;   /* tid % steps */

    /* Skip invalid steps */
    if (step >= valid[chain])
        return;

    /* H3: steps is power of 2, use shift for index computation */
    int idx = (chain << steps_log2) + step;
    const uint8_t *base_priv = base_privkeys + chain * 32;
    const uint8_t *h160c = hash160_comp + idx * 20;
    const uint8_t *h160uc = hash160_uncomp + idx * 20;

    /* Look up compressed pubkey Hash160 */
    uint32_t fp_c = ((uint32_t)h160c[0] << 24) | ((uint32_t)h160c[1] << 16) |
                    ((uint32_t)h160c[2] <<  8) | ((uint32_t)h160c[3]);
    uint32_t slot_c = fp_c & ht_mask;
    for (;;) {
        /* H2: __ldg routes through read-only cache; 32-byte aligned slot = 1 cache line */
        uint32_t sfp = __ldg(&ht_slots[slot_c].fp);
        if (sfp == 0)
            break;  /* empty slot, no hit */
        if (sfp == fp_c) {
            /* Fingerprint match, verify full 20 bytes */
            int match = 1;
            for (int k = 0; k < 20; k++) {
                if (__ldg(&ht_slots[slot_c].h160[k]) != h160c[k]) {
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
        uint32_t sfp = __ldg(&ht_slots[slot_uc].fp);
        if (sfp == 0)
            break;
        if (sfp == fp_uc) {
            int match = 1;
            for (int k = 0; k < 20; k++) {
                if (__ldg(&ht_slots[slot_uc].h160[k]) != h160uc[k]) {
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

    /*
     * H2: gpu_ht_slot is 32 bytes (padded), but CPU-side ht_slot is 24 bytes.
     * Direct cudaMemcpy would corrupt data; convert slot-by-slot via a
     * pinned host staging buffer to preserve correct layout.
     */
    struct gpu_ht_slot *h_staging = NULL;
    if (cudaMallocHost(&h_staging, ht_size) != cudaSuccess) {
        keylog_warn("[GPU] Hash table staging buffer allocation failed, falling back to CPU-side lookup");
        cudaFree(d_ht_slots);
        d_ht_slots = NULL;
        g_use_gpu_ht = 0;
        return 0;
    }
    for (uint32_t i = 0; i < ht_capacity; i++) {
        h_staging[i].fp = ht_slots_cpu[i].fp;
        memcpy(h_staging[i].h160, ht_slots_cpu[i].h160, 20);
        memset(h_staging[i]._pad, 0, 8);
    }

    /* Upload converted hash table (once, remains resident throughout search) */
    if (cudaMemcpy(d_ht_slots, h_staging, ht_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        keylog_error("[GPU] Hash table upload failed");
        cudaFreeHost(h_staging);
        cudaFree(d_ht_slots);
        d_ht_slots = NULL;
        return -1;
    }
    cudaFreeHost(h_staging);

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
 * Tiling granularity: each thread processes HASH160_TILE consecutive steps.
 * HASH160_TILE=16 gives total_threads ≈ 4M, which saturates the GPU while
 * keeping per-thread local memory to one sha256/ripemd160 working set.
 */
#define HASH160_TILE 16

__global__ void kernel_hash160_lookup(
    const uint8_t            * __restrict__ aff_x,
    const uint8_t            * __restrict__ aff_y,
    const int                * __restrict__ valid,
    const uint8_t            * __restrict__ base_privkeys,
    const struct gpu_ht_slot * __restrict__ ht_slots,
    uint32_t ht_mask,
    gpu_hit_result_t         * __restrict__ hit_results,
    int                      * __restrict__ hit_count,
    int num_chains,
    int steps,
    int tiles_per_chain)   /* = steps / HASH160_TILE */
{
    /* Each thread handles one tile: (chain, tile_idx) */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tiles = num_chains * tiles_per_chain;
    if (tid >= total_tiles)
        return;

    int chain      = tid / tiles_per_chain;
    int tile_idx   = tid % tiles_per_chain;
    int step_start = tile_idx * HASH160_TILE;
    int step_end   = step_start + HASH160_TILE;

    int chain_valid = valid[chain];
    if (step_start >= chain_valid)
        return;
    if (step_end > chain_valid)
        step_end = chain_valid;

    const uint8_t *base_priv = base_privkeys + chain * 32;

    /* Reuse these buffers across the tile loop to avoid repeated local alloc */
    uint8_t comp_block[64];
    uint8_t uncomp_block1[64];
    uint8_t uncomp_block2[64];
    uint8_t sha256_out[32];
    uint8_t h160c[20];
    uint8_t h160uc[20];

    for (int step = step_start; step < step_end; step++) {
        /* step-major layout: [step * num_chains + chain] → coalesced reads across warp */
        int idx = step * num_chains + chain;
        const uint8_t *x = aff_x + idx * 32;
        const uint8_t *y = aff_y + idx * 32;

        /* ---- Compressed pubkey Hash160 ---- */
        comp_block[0] = (y[31] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++)
            comp_block[1 + i] = x[i];
        comp_block[33] = 0x80;
        for (int i = 34; i < 56; i++)
            comp_block[i] = 0;
        comp_block[56] = 0; comp_block[57] = 0; comp_block[58] = 0; comp_block[59] = 0;
        comp_block[60] = 0; comp_block[61] = 0; comp_block[62] = 0x01; comp_block[63] = 0x08;

        sha256_single_block(comp_block, sha256_out);
        ripemd160(sha256_out, h160c);

        /* Lookup: compressed address */
        uint32_t fp_c = ((uint32_t)h160c[0] << 24) | ((uint32_t)h160c[1] << 16) |
                        ((uint32_t)h160c[2] <<  8) | ((uint32_t)h160c[3]);
        uint32_t slot_c = fp_c & ht_mask;
        for (;;) {
            uint32_t sfp = __ldg(&ht_slots[slot_c].fp);
            if (sfp == 0) break;
            if (sfp == fp_c) {
                int match = 1;
                for (int k = 0; k < 20; k++) {
                    if (__ldg(&ht_slots[slot_c].h160[k]) != h160c[k]) { match = 0; break; }
                }
                if (match) {
                    int pos = atomicAdd(hit_count, 1);
                    if (pos < GPU_MAX_HITS) {
                        gpu_hit_result_t *r = &hit_results[pos];
                        for (int k = 0; k < 32; k++) r->privkey[k] = base_priv[k];
                        r->privkey[28] = (uint8_t)(step >> 24);
                        r->privkey[29] = (uint8_t)(step >> 16);
                        r->privkey[30] = (uint8_t)(step >>  8);
                        r->privkey[31] = (uint8_t)(step);
                        for (int k = 0; k < 20; k++) r->hash160[k] = h160c[k];
                        r->is_compressed = 1;
                    }
                    break;
                }
            }
            slot_c = (slot_c + 1) & ht_mask;
        }

        /* ---- Uncompressed pubkey Hash160 ---- */
        uncomp_block1[0] = 0x04;
        for (int i = 0; i < 32; i++) uncomp_block1[1 + i] = x[i];
        for (int i = 0; i < 31; i++) uncomp_block1[33 + i] = y[i];
        uncomp_block2[0] = y[31];
        uncomp_block2[1] = 0x80;
        for (int i = 2; i < 56; i++) uncomp_block2[i] = 0;
        uncomp_block2[56] = 0; uncomp_block2[57] = 0; uncomp_block2[58] = 0; uncomp_block2[59] = 0;
        uncomp_block2[60] = 0; uncomp_block2[61] = 0; uncomp_block2[62] = 0x02; uncomp_block2[63] = 0x08;

        sha256_two_blocks(uncomp_block1, uncomp_block2, sha256_out);
        ripemd160(sha256_out, h160uc);

        /* Lookup: uncompressed address */
        uint32_t fp_uc = ((uint32_t)h160uc[0] << 24) | ((uint32_t)h160uc[1] << 16) |
                         ((uint32_t)h160uc[2] <<  8) | ((uint32_t)h160uc[3]);
        uint32_t slot_uc = fp_uc & ht_mask;
        for (;;) {
            uint32_t sfp = __ldg(&ht_slots[slot_uc].fp);
            if (sfp == 0) break;
            if (sfp == fp_uc) {
                int match = 1;
                for (int k = 0; k < 20; k++) {
                    if (__ldg(&ht_slots[slot_uc].h160[k]) != h160uc[k]) { match = 0; break; }
                }
                if (match) {
                    int pos = atomicAdd(hit_count, 1);
                    if (pos < GPU_MAX_HITS) {
                        gpu_hit_result_t *r = &hit_results[pos];
                        for (int k = 0; k < 32; k++) r->privkey[k] = base_priv[k];
                        r->privkey[28] = (uint8_t)(step >> 24);
                        r->privkey[29] = (uint8_t)(step >> 16);
                        r->privkey[30] = (uint8_t)(step >>  8);
                        r->privkey[31] = (uint8_t)(step);
                        for (int k = 0; k < 20; k++) r->hash160[k] = h160uc[k];
                        r->is_compressed = 0;
                    }
                    break;
                }
            }
            slot_uc = (slot_uc + 1) & ht_mask;
        }
    } /* end tile loop */
}

/*
 * Execute fused Hash160+lookup kernel (GPU mode only, no hash160 written to VRAM).
 * aff_x, aff_y, valid, base_privkeys: device pointers from gpu_secp256k1.cu
 * Returns hit count, or -1 on error.
 */
int gpu_hash160_lookup_run(
    const uint8_t *d_aff_x,
    const uint8_t *d_aff_y,
    const int     *d_valid,
    const uint8_t *d_base_privkeys,
    int num_chains, int steps,
    gpu_hit_result_t *h_hits, int *h_hit_count,
    cudaStream_t stream)
{
    if (!g_use_gpu_ht)
        return 0;  /* not in GPU mode, skip */

    cudaMemsetAsync(d_hit_count, 0, sizeof(int), stream);

    /* One thread per tile (HASH160_TILE steps); total_threads = num_chains * tiles_per_chain */
    int tiles_per_chain = (steps + HASH160_TILE - 1) / HASH160_TILE;
    int block_size = 256;
    int total_tiles = num_chains * tiles_per_chain;
    int grid_size = (total_tiles + block_size - 1) / block_size;

    kernel_hash160_lookup<<<grid_size, block_size, 0, stream>>>(
        d_aff_x, d_aff_y,
        d_valid, d_base_privkeys,
        d_ht_slots, d_ht_mask_val,
        d_hit_results, d_hit_count,
        num_chains, steps, tiles_per_chain);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        keylog_error("[GPU] kernel_hash160_lookup launch failed: %s", cudaGetErrorString(err));
        return -1;
    }

    cudaStreamSynchronize(stream);

    int hit_count = 0;
    cudaMemcpy(&hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    *h_hit_count = hit_count;

    if (hit_count > 0) {
        int actual = (hit_count > GPU_MAX_HITS) ? GPU_MAX_HITS : hit_count;
        cudaMemcpy(h_hits, d_hit_results, actual * sizeof(gpu_hit_result_t), cudaMemcpyDeviceToHost);
    }

    return hit_count;
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

    /*
     * H2+H3: steps must be power of 2; compute steps_log2 for bit-shift indexing.
     * Total threads = num_chains * steps (unchanged from original layout).
     */
    int steps_log2 = 0;
    { int s = steps; while (s > 1) { steps_log2++; s >>= 1; } }

    int block_size = 256;
    int total_threads = num_chains * steps;
    int grid_size = (total_threads + block_size - 1) / block_size;

    kernel_hashtable_lookup<<<grid_size, block_size, 0, stream>>>(
        d_hash160_comp, d_hash160_uncomp,
        d_valid, d_base_privkeys,
        d_ht_slots, d_ht_mask_val,
        d_hit_results, d_hit_count,
        num_chains, steps, steps_log2);

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
