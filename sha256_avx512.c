/*
 * sha256_avx512.c — SHA256 16-way AVX-512 parallel compression function
 *
 * Processes 16 independent message blocks (each 64 bytes) simultaneously,
 * using AVX-512 512-bit registers to pack 16 uint32_t values into one __m512i,
 * advancing 16 lanes of state with each SIMD instruction.
 *
 * Interface:
 *   void sha256_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16])
 */

#ifdef __AVX512F__

#include <immintrin.h>
#include <stdint.h>
#include "sha256.h"
#include "avx512_common.h"

/* Right rotation */
#define V_ROR(x, n) _mm512_or_si512(_mm512_srli_epi32((x), (n)), _mm512_slli_epi32((x), 32 - (n)))

/*
 * SHA256 auxiliary functions (vectorized, single-instruction via AVX-512 ternarylogic)
 *
 * V_CH(x,y,z) = (x & y) ^ (~x & z)  => ternarylogic imm8 = 0xCA
 * V_MAJ(x,y,z) = (x & y) ^ (x & z) ^ (y & z) => ternarylogic imm8 = 0xE8
 */
#define V_CH(x, y, z)   _mm512_ternarylogic_epi32((x), (y), (z), 0xCA)
#define V_MAJ(x, y, z)  _mm512_ternarylogic_epi32((x), (y), (z), 0xE8)
#define V_EP0(x)        _mm512_xor_si512(_mm512_xor_si512(V_ROR((x), 2), V_ROR((x), 13)), V_ROR((x), 22))
#define V_EP1(x)        _mm512_xor_si512(_mm512_xor_si512(V_ROR((x), 6), V_ROR((x), 11)), V_ROR((x), 25))
#define V_SIG0(x)       _mm512_xor_si512(_mm512_xor_si512(V_ROR((x), 7), V_ROR((x), 18)), _mm512_srli_epi32((x), 3))
#define V_SIG1(x)       _mm512_xor_si512(_mm512_xor_si512(V_ROR((x), 17), V_ROR((x), 19)), _mm512_srli_epi32((x), 10))

/* One SHA256 round (vectorized): advances 16 lanes of state simultaneously */
#define V_ROUND(a, b, c, d, e, f, g, h, k_val, w)                                           \
    do {                                                                                    \
        __m512i _k = _mm512_set1_epi32(k_val);                                              \
        __m512i _t1 = _mm512_add_epi32(                                                     \
            _mm512_add_epi32(_mm512_add_epi32((h), V_EP1(e)),                               \
            _mm512_add_epi32(V_CH(e, f, g), _k)), (w));                                     \
        __m512i _t2 = _mm512_add_epi32(V_EP0(a), V_MAJ(a, b, c));                           \
        (d) = _mm512_add_epi32((d), _t1);                                                   \
        (h) = _mm512_add_epi32(_t1, _t2);                                                   \
    } while (0)

/* Message expansion */
#define V_EXPAND(w0, w1, w9, w14) \
    _mm512_add_epi32(_mm512_add_epi32(V_SIG1(w14), (w9)), _mm512_add_epi32(V_SIG0(w1), (w0)))

/* Load the i-th uint32_t (big-endian) from each of 16 blocks into a __m512i */
static inline __attribute__((always_inline)) __m512i load_be32_16way(const uint8_t *const blocks[16], int i)
{
    const int off = i * 4;
    uint32_t lanes[16] __attribute__((aligned(64))) = {
        __builtin_bswap32(load_u32_unaligned(blocks[0] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[1] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[2] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[3] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[4] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[5] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[6] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[7] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[8] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[9] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[10] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[11] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[12] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[13] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[14] + off)),
        __builtin_bswap32(load_u32_unaligned(blocks[15] + off))
    };
    return _mm512_load_si512((const void *)lanes);
}

/* Write the 16 lanes of a __m512i back to the i-th element of each of 16 states */
static inline void store_16way(uint32_t *const states[16], int i, __m512i v)
{
    uint32_t tmp[16] __attribute__((aligned(64)));
    _mm512_store_si512((__m512i *)tmp, v);
    states[0][i] = tmp[0];
    states[1][i] = tmp[1];
    states[2][i] = tmp[2];
    states[3][i] = tmp[3];
    states[4][i] = tmp[4];
    states[5][i] = tmp[5];
    states[6][i] = tmp[6];
    states[7][i] = tmp[7];
    states[8][i] = tmp[8];
    states[9][i] = tmp[9];
    states[10][i] = tmp[10];
    states[11][i] = tmp[11];
    states[12][i] = tmp[12];
    states[13][i] = tmp[13];
    states[14][i] = tmp[14];
    states[15][i] = tmp[15];
}

/*
 * Perform one SHA256 compression on 16 independent (state, block) pairs simultaneously
 */
__attribute__((target("avx512f")))
void sha256_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16])
{
    /* Load 16-lane initial state */
    __m512i a = load_state_word_16way(states, 0);
    __m512i b = load_state_word_16way(states, 1);
    __m512i c = load_state_word_16way(states, 2);
    __m512i d = load_state_word_16way(states, 3);
    __m512i e = load_state_word_16way(states, 4);
    __m512i f = load_state_word_16way(states, 5);
    __m512i g = load_state_word_16way(states, 6);
    __m512i h = load_state_word_16way(states, 7);

    /* Save initial state for final accumulation */
    __m512i a0 = a, b0 = b, c0 = c, d0 = d, e0 = e, f0 = f, g0 = g, h0 = h;

    /* Load 16-lane message words (big-endian) */
    __m512i w0 = load_be32_16way(blocks, 0), w1 = load_be32_16way(blocks, 1);
    __m512i w2 = load_be32_16way(blocks, 2), w3 = load_be32_16way(blocks, 3);
    __m512i w4 = load_be32_16way(blocks, 4), w5 = load_be32_16way(blocks, 5);
    __m512i w6 = load_be32_16way(blocks, 6), w7 = load_be32_16way(blocks, 7);
    __m512i w8 = load_be32_16way(blocks, 8), w9 = load_be32_16way(blocks, 9);
    __m512i w10 = load_be32_16way(blocks, 10), w11 = load_be32_16way(blocks, 11);
    __m512i w12 = load_be32_16way(blocks, 12), w13 = load_be32_16way(blocks, 13);
    __m512i w14 = load_be32_16way(blocks, 14), w15 = load_be32_16way(blocks, 15);

    /* Rounds 0-15 */
    V_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98, w0);
    V_ROUND(h, a, b, c, d, e, f, g, 0x71374491, w1);
    V_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcf, w2);
    V_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba5, w3);
    V_ROUND(e, f, g, h, a, b, c, d, 0x3956c25b, w4);
    V_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1, w5);
    V_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4, w6);
    V_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5, w7);
    V_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98, w8);
    V_ROUND(h, a, b, c, d, e, f, g, 0x12835b01, w9);
    V_ROUND(g, h, a, b, c, d, e, f, 0x243185be, w10);
    V_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3, w11);
    V_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74, w12);
    V_ROUND(d, e, f, g, h, a, b, c, 0x80deb1fe, w13);
    V_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a7, w14);
    V_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174, w15);

    /* Rounds 16-31 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c1, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc6, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x240ca1cc, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6f, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aa, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x76f988da, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0x983e5152, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0xa831c66d, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0xb00327c8, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf3, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0x14292967, w15);

    /* Rounds 32-47 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0x27b70a85, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0x2e1b2138, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x53380d13, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x650a7354, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x766a0abb, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92e, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x92722c85, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0xa81a664b, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a3, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0xd192e819, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xd6990624, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0xf40e3585, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0x106aa070, w15);

    /* Rounds 48-63 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0x1e376c08, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x2748774c, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4f, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0x748f82ee, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0x78a5636f, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0x84c87814, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0x8cc70208, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0x90befffa, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xa4506ceb, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2, w15);

    /* Accumulate initial state */
    a = _mm512_add_epi32(a, a0);
    b = _mm512_add_epi32(b, b0);
    c = _mm512_add_epi32(c, c0);
    d = _mm512_add_epi32(d, d0);
    e = _mm512_add_epi32(e, e0);
    f = _mm512_add_epi32(f, f0);
    g = _mm512_add_epi32(g, g0);
    h = _mm512_add_epi32(h, h0);

    /* Write back 16-lane state */
    store_16way(states, 0, a);
    store_16way(states, 1, b);
    store_16way(states, 2, c);
    store_16way(states, 3, d);
    store_16way(states, 4, e);
    store_16way(states, 5, f);
    store_16way(states, 6, g);
    store_16way(states, 7, h);
}

/*
 * sha256_compress_avx512_contig — gather-optimized variant for contiguous block arrays.
 *
 * When the 16 message blocks reside in a contiguous array (e.g. blocks[16][64]),
 * this function uses _mm512_i32gather_epi32 to collect each message word from all
 * 16 blocks in a single instruction, replacing the 16-scalar-load + aligned-buffer
 * + _mm512_load_si512 path used by load_be32_16way.
 *
 * Parameters:
 *   states : array of 16 pointers to uint32_t[8] state arrays
 *   base   : pointer to blocks[0][0] (start of contiguous block memory)
 *   stride : byte distance between consecutive blocks (e.g. 64)
 */
__attribute__((target("avx512f,avx512bw")))
void sha256_compress_avx512_contig(uint32_t *states[16], const uint8_t *base, int stride)
{
    /* Load 16-lane initial state */
    __m512i a = load_state_word_16way(states, 0);
    __m512i b = load_state_word_16way(states, 1);
    __m512i c = load_state_word_16way(states, 2);
    __m512i d = load_state_word_16way(states, 3);
    __m512i e = load_state_word_16way(states, 4);
    __m512i f = load_state_word_16way(states, 5);
    __m512i g = load_state_word_16way(states, 6);
    __m512i h = load_state_word_16way(states, 7);

    /* Save initial state for final accumulation */
    __m512i a0 = a, b0 = b, c0 = c, d0 = d, e0 = e, f0 = f, g0 = g, h0 = h;

    /* Load 16-lane message words using gather (big-endian) */
    __m512i w0 = load_be32_contig(base, stride, 0), w1 = load_be32_contig(base, stride, 1);
    __m512i w2 = load_be32_contig(base, stride, 2), w3 = load_be32_contig(base, stride, 3);
    __m512i w4 = load_be32_contig(base, stride, 4), w5 = load_be32_contig(base, stride, 5);
    __m512i w6 = load_be32_contig(base, stride, 6), w7 = load_be32_contig(base, stride, 7);
    __m512i w8 = load_be32_contig(base, stride, 8), w9 = load_be32_contig(base, stride, 9);
    __m512i w10 = load_be32_contig(base, stride, 10), w11 = load_be32_contig(base, stride, 11);
    __m512i w12 = load_be32_contig(base, stride, 12), w13 = load_be32_contig(base, stride, 13);
    __m512i w14 = load_be32_contig(base, stride, 14), w15 = load_be32_contig(base, stride, 15);

    /* Rounds 0-15 */
    V_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98, w0);
    V_ROUND(h, a, b, c, d, e, f, g, 0x71374491, w1);
    V_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcf, w2);
    V_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba5, w3);
    V_ROUND(e, f, g, h, a, b, c, d, 0x3956c25b, w4);
    V_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1, w5);
    V_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4, w6);
    V_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5, w7);
    V_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98, w8);
    V_ROUND(h, a, b, c, d, e, f, g, 0x12835b01, w9);
    V_ROUND(g, h, a, b, c, d, e, f, 0x243185be, w10);
    V_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3, w11);
    V_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74, w12);
    V_ROUND(d, e, f, g, h, a, b, c, 0x80deb1fe, w13);
    V_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a7, w14);
    V_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174, w15);

    /* Rounds 16-31 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c1, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc6, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x240ca1cc, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6f, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aa, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x76f988da, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0x983e5152, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0xa831c66d, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0xb00327c8, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf3, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0x14292967, w15);

    /* Rounds 32-47 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0x27b70a85, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0x2e1b2138, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x53380d13, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x650a7354, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x766a0abb, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92e, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x92722c85, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0xa81a664b, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a3, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0xd192e819, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xd6990624, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0xf40e3585, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0x106aa070, w15);

    /* Rounds 48-63 */
    w0 = V_EXPAND(w0, w1, w9, w14);
    V_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116, w0);
    w1 = V_EXPAND(w1, w2, w10, w15);
    V_ROUND(h, a, b, c, d, e, f, g, 0x1e376c08, w1);
    w2 = V_EXPAND(w2, w3, w11, w0);
    V_ROUND(g, h, a, b, c, d, e, f, 0x2748774c, w2);
    w3 = V_EXPAND(w3, w4, w12, w1);
    V_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5, w3);
    w4 = V_EXPAND(w4, w5, w13, w2);
    V_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3, w4);
    w5 = V_EXPAND(w5, w6, w14, w3);
    V_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w5);
    w6 = V_EXPAND(w6, w7, w15, w4);
    V_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4f, w6);
    w7 = V_EXPAND(w7, w8, w0, w5);
    V_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3, w7);
    w8 = V_EXPAND(w8, w9, w1, w6);
    V_ROUND(a, b, c, d, e, f, g, h, 0x748f82ee, w8);
    w9 = V_EXPAND(w9, w10, w2, w7);
    V_ROUND(h, a, b, c, d, e, f, g, 0x78a5636f, w9);
    w10 = V_EXPAND(w10, w11, w3, w8);
    V_ROUND(g, h, a, b, c, d, e, f, 0x84c87814, w10);
    w11 = V_EXPAND(w11, w12, w4, w9);
    V_ROUND(f, g, h, a, b, c, d, e, 0x8cc70208, w11);
    w12 = V_EXPAND(w12, w13, w5, w10);
    V_ROUND(e, f, g, h, a, b, c, d, 0x90befffa, w12);
    w13 = V_EXPAND(w13, w14, w6, w11);
    V_ROUND(d, e, f, g, h, a, b, c, 0xa4506ceb, w13);
    w14 = V_EXPAND(w14, w15, w7, w12);
    V_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7, w14);
    w15 = V_EXPAND(w15, w0, w8, w13);
    V_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2, w15);

    /* Accumulate initial state */
    a = _mm512_add_epi32(a, a0);
    b = _mm512_add_epi32(b, b0);
    c = _mm512_add_epi32(c, c0);
    d = _mm512_add_epi32(d, d0);
    e = _mm512_add_epi32(e, e0);
    f = _mm512_add_epi32(f, f0);
    g = _mm512_add_epi32(g, g0);
    h = _mm512_add_epi32(h, h0);

    /* Write back 16-lane state */
    store_16way(states, 0, a);
    store_16way(states, 1, b);
    store_16way(states, 2, c);
    store_16way(states, 3, d);
    store_16way(states, 4, e);
    store_16way(states, 5, f);
    store_16way(states, 6, g);
    store_16way(states, 7, h);
}

#endif /* __AVX512F__ */
