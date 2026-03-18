/*
 * avx512_common.h — Shared inline utilities for AVX-512 16-way hash functions
 *
 * Contains common load/store helpers used by both sha256_avx512.c and
 * ripemd160_avx512.c.  All functions are static inline to avoid link-time
 * symbol conflicts across compilation units.
 *
 * Requires: AVX-512F (-mavx512f or -march=native on a capable CPU).
 */

#ifndef AVX512_COMMON_H
#define AVX512_COMMON_H

#ifdef __AVX512F__

#include <immintrin.h>
#include <stdint.h>

/*
 * Load a possibly-unaligned uint32_t via __builtin_memcpy.
 * The compiler will emit a single mov (or movbe) instruction on x86.
 */
static inline __attribute__((always_inline)) uint32_t load_u32_unaligned(const uint8_t *p)
{
    uint32_t v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}

#endif /* __AVX512F__ */
#endif /* AVX512_COMMON_H */

