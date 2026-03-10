/*
 * gpu/gpu_detect.cu
 *
 * Runtime GPU detection and selection:
 *   - Enumerate all available CUDA devices
 *   - Print name, VRAM, SM count, compute capability for each GPU
 *   - Print performance warning if compute capability < 6.0
 *   - Select device with most VRAM when multiple GPUs present
 *   - Check VRAM sufficiency, reduce batch size if insufficient
 *
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gpu_interface.h"
#include "../keylog.h"

/* Currently selected device info (global, valid after gpu_init) */
static gpu_device_info_t g_device_info;

/* Actual batch size in use (may be reduced due to insufficient VRAM) */
static int g_actual_batch_size = GPU_BATCH_SIZE;

/* Internal Helper Functions */

/* Format byte count as human-readable string (MB/GB) */
static void format_mem(size_t bytes, char *buf, size_t buf_size)
{
    if (bytes >= (size_t)(1024 * 1024 * 1024))
        snprintf(buf, buf_size, "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    else
        snprintf(buf, buf_size, "%.0f MB", (double)bytes / (1024.0 * 1024.0));
}

/*
 * Estimate VRAM required per batch (bytes):
 *   - GPU work buffer (pubkey affine coordinates): batch_size * 64 bytes (X+Y each 32 bytes)
 *   - Hash160 result buffer: batch_size * 2 * 20 bytes (compressed + uncompressed)
 *   - Hit result buffer: GPU_MAX_HITS * sizeof(gpu_hit_result_t)
 *   - Hash table: ht_capacity * 24 bytes (fp 4 + h160 20)
 * Note: secp256k1 intermediate Jacobian coordinate buffers managed separately in gpu_secp256k1.cu
 */
static size_t estimate_gpu_mem(int batch_size, uint32_t ht_capacity)
{
    size_t pubkey_buf = (size_t)batch_size * 64;
    size_t hash160_buf = (size_t)batch_size * 2 * 20;
    size_t hit_buf = GPU_MAX_HITS * sizeof(gpu_hit_result_t);
    size_t ht_buf = (size_t)ht_capacity * 24;
    /* Reserve 128MB for CUDA runtime and driver */
    size_t overhead = 128ULL * 1024 * 1024;
    return pubkey_buf + hash160_buf + hit_buf + ht_buf + overhead;
}

/* External Interface */

/*
 * gpu_detect_and_select: enumerate devices, select best device, check VRAM
 * ht_capacity : hash table capacity (for VRAM estimation)
 * out_batch   : output actual usable batch size
 * Return value: 0 success (g_device_info filled), -1 failure
 */
int gpu_detect_and_select(uint32_t ht_capacity, int *out_batch)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        keylog_error("[GPU] No available CUDA devices detected: %s",
                     cudaGetErrorString(err));
        return -1;
    }

    keylog_info("[GPU] Detected %d CUDA device(s):", device_count);

    int best_id   = 0;
    size_t best_mem = 0;

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        size_t free_mem = 0, total_mem = 0;
        /* Temporarily switch to this device to query available VRAM */
        cudaSetDevice(i);
        cudaMemGetInfo(&free_mem, &total_mem);

        char total_str[32], free_str[32];
        format_mem(total_mem, total_str, sizeof(total_str));
        format_mem(free_mem, free_str, sizeof(free_str));

        keylog_info("[GPU]   Device %d: %s | VRAM: %s (free %s) | SM: %d | Compute: %d.%d",
            i, prop.name, total_str, free_str, prop.multiProcessorCount, prop.major, prop.minor);

        /* Compute capability < 6.0 warning */
        if (prop.major < 6) {
            keylog_warn("[GPU]   Device %d compute capability %d.%d < 6.0 (Pascal), performance may be poor",
                i, prop.major, prop.minor);
        }

        /* Select device with most total VRAM (CUDA_VISIBLE_DEVICES already filtered at driver level) */
        if (total_mem > best_mem) {
            best_mem = total_mem;
            best_id = i;
        }
    }

    /* Switch to best device */
    cudaSetDevice(best_id);

    cudaDeviceProp best_prop;
    cudaGetDeviceProperties(&best_prop, best_id);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    /* Fill device info */
    memset(&g_device_info, 0, sizeof(g_device_info));
    g_device_info.device_id = best_id;
    strncpy(g_device_info.name, best_prop.name, sizeof(g_device_info.name) - 1);
    g_device_info.total_mem = total_mem;
    g_device_info.free_mem = free_mem;
    g_device_info.sm_count = best_prop.multiProcessorCount;
    g_device_info.cc_major = best_prop.major;
    g_device_info.cc_minor = best_prop.minor;
    g_device_info.max_threads_per_block = best_prop.maxThreadsPerBlock;
    g_device_info.warp_size = best_prop.warpSize;

    keylog_info("[GPU] Selected device %d: %s (most VRAM)", best_id, best_prop.name);

    /* Check VRAM sufficiency, progressively reduce batch size if insufficient */
    int batch = GPU_BATCH_SIZE;
    while (batch >= 1024) {
        size_t needed = estimate_gpu_mem(batch, ht_capacity);
        if (needed <= free_mem) {
            break;
        }
        char need_str[32], free_str[32];
        format_mem(needed, need_str, sizeof(need_str));
        format_mem(free_mem, free_str, sizeof(free_str));
        keylog_warn("[GPU] Batch size %d requires %s, available %s, trying to reduce batch size",
            batch, need_str, free_str);
        batch >>= 1;  /* halve */
    }

    if (batch < 1024) {
        keylog_error("[GPU] Severely insufficient VRAM, cannot allocate minimum work buffer, exiting");
        return -1;
    }

    if (batch != GPU_BATCH_SIZE) {
        keylog_warn("[GPU] Batch size reduced from %d to %d", GPU_BATCH_SIZE, batch);
    }

    g_actual_batch_size = batch;
    *out_batch = batch;

    char total_str[32], free_str[32];
    format_mem(total_mem, total_str, sizeof(total_str));
    format_mem(free_mem,  free_str,  sizeof(free_str));
    keylog_info("[GPU] Device info: SM=%d | Compute=%d.%d | VRAM=%s (free %s) | batch=%d",
        g_device_info.sm_count, g_device_info.cc_major, g_device_info.cc_minor,
        total_str, free_str, batch);

    return 0;
}

/* Return currently selected device info */
const gpu_device_info_t *gpu_get_device_info(void)
{
    return &g_device_info;
}

/* Return actual batch size */
int gpu_get_actual_batch_size(void)
{
    return g_actual_batch_size;
}
