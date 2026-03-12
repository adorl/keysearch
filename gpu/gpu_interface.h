#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

/*
 * gpu_interface.h
 *
 * Common interface definitions between CPU and GPU modules.
 * Only included when the USE_GPU macro is defined.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Number of chains processed per GPU batch (must be power of 2, auto-adjusted based on VRAM) */
#define GPU_BATCH_SIZE      (1 << 18)    /* default 256K chains */

/* Incremental steps per chain (steps derived from base private key per chain) */
/* H3: must be a power of 2; kernel_hashtable_lookup uses bit-shift/AND instead of division */
#define GPU_CHAIN_STEPS     (256)        /* 256 steps per chain (2^8) */

/* Hit result buffer size (max hits returned per batch) */
#define GPU_MAX_HITS        (64)


typedef struct {
    int     device_id;              /* CUDA device index */
    char    name[256];              /* device name */
    size_t  total_mem;              /* total VRAM (bytes) */
    size_t  free_mem;               /* available VRAM (bytes) */
    int     sm_count;               /* SM count */
    int     cc_major;               /* compute capability major version */
    int     cc_minor;               /* compute capability minor version */
    int     max_threads_per_block;  /* max threads per block */
    int     warp_size;              /* warp size */
} gpu_device_info_t;

/* Hit Result Struct */
typedef struct {
    uint8_t privkey[32];            /* hit private key (32 bytes) */
    uint8_t hash160[20];            /* hit Hash160 (20 bytes) */
    int     is_compressed;          /* 1=compressed address hit, 0=uncompressed address hit */
} gpu_hit_result_t;

/* Interface Function Declarations */

/*
 * Initialize GPU: detect device, select best device, allocate VRAM buffers, upload hash table
 * ht_slots    : hash table slot array (provided by ht_slots in hash_utils.h)
 * ht_capacity : hash table capacity (slot count, must be power of 2)
 * Return value: 0 success, -1 failure
 */
int gpu_init(const void *ht_slots, uint32_t ht_capacity);

/*
 * Start GPU search main loop (blocks until search ends)
 * Return value: 0 normal end, -1 error
 */
int gpu_search(void);

/*
 * Free GPU resources (VRAM, context, etc.)
 */
void gpu_cleanup(void);

/*
 * Get currently selected GPU device info (available after gpu_init)
 * Returns internal static pointer, caller must not free
 */
const gpu_device_info_t *gpu_get_device_info(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_INTERFACE_H */
