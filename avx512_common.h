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

/*
 * Gather-based load for contiguous block arrays (e.g. blocks[16][64]).
 *
 * When the 16 blocks reside in a contiguous 2D array with a fixed stride,
 * we can use _mm512_i32gather_epi32 to collect the i-th uint32_t from all
 * 16 blocks in a single instruction, replacing the 16-scalar-load + aligned
 * buffer + _mm512_load_si512 pattern.
 *
 * Parameters:
 *   base   : pointer to the start of blocks[0]
 *   stride : byte distance between consecutive blocks (e.g. 64 for [16][64])
 *   i      : word index within each block (0..15 for a 64-byte block)
 *
 * The index vector contains byte offsets: lane_k = k * stride + i * 4
 */

/*
 * Build a __m512i index vector for 16-lane gather with given stride and word offset.
 * Each lane k gets byte offset: k * stride + word_idx * 4
 */
static inline __attribute__((always_inline)) __m512i
gather_index_16way(int stride, int word_idx)
{
    int off = word_idx * 4;
    return _mm512_set_epi32(
        15 * stride + off, 14 * stride + off, 13 * stride + off, 12 * stride + off,
        11 * stride + off, 10 * stride + off,  9 * stride + off,  8 * stride + off,
         7 * stride + off,  6 * stride + off,  5 * stride + off,  4 * stride + off,
         3 * stride + off,  2 * stride + off,  1 * stride + off,  0 * stride + off);
}

/*
 * Load the i-th big-endian uint32_t from 16 contiguous blocks using gather.
 * Performs gather + in-register byte-swap via _mm512_shuffle_epi8 (requires AVX-512BW).
 */
static inline __attribute__((always_inline)) __m512i
load_be32_contig(const uint8_t *base, int stride, int i)
{
    __m512i idx = gather_index_16way(stride, i);
    __m512i raw = _mm512_i32gather_epi32(idx, (const int *)base, 1);
    /* Byte-swap each 32-bit lane: big-endian -> native (little-endian on x86) */
    const __m512i bswap_mask = _mm512_set_epi8(
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3);
    return _mm512_shuffle_epi8(raw, bswap_mask);
}

/*
 * Load the i-th little-endian uint32_t from 16 contiguous blocks using gather.
 * No byte-swap needed on x86 (native LE).
 */
static inline __attribute__((always_inline)) __m512i
load_le32_contig(const uint8_t *base, int stride, int i)
{
    __m512i idx = gather_index_16way(stride, i);
    return _mm512_i32gather_epi32(idx, (const int *)base, 1);
}

#endif /* __AVX512F__ */
#endif /* AVX512_COMMON_H */
