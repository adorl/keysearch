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

/*
 * Gather the i-th uint32_t element from each of 16 states into a __m512i.
 * Uses an aligned stack buffer + _mm512_load_si512 instead of the slower
 * _mm512_set_epi32 intrinsic (which generates 16 scalar insert instructions).
 */
static inline __attribute__((always_inline)) __m512i load_state_word_16way(uint32_t *const states[16], int i)
{
    uint32_t lanes[16] __attribute__((aligned(64))) = {
        states[0][i], states[1][i], states[2][i], states[3][i],
        states[4][i], states[5][i], states[6][i], states[7][i],
        states[8][i], states[9][i], states[10][i], states[11][i],
        states[12][i], states[13][i], states[14][i], states[15][i]
    };
    return _mm512_load_si512((const void *)lanes);
}

#endif /* __AVX512F__ */
#endif /* AVX512_COMMON_H */
