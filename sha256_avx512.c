/*
 * sha256_avx512.c — SHA256 16-way AVX-512 parallel compression function (SoA variants)
 *
 * Processes 16 independent message blocks (each 64 bytes) simultaneously,
 * using AVX-512 512-bit registers to pack 16 uint32_t values into one __m512i,
 * advancing 16 lanes of state with each SIMD instruction.
 *
 * SoA Interface:
 *   void sha256_compress_avx512_soa(__m512i soa_state[8], const uint8_t *blocks[16])
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

/*
 * sha256_compress_avx512_soa — SoA (Structure-of-Arrays) variant.
 *
 * Instead of reading/writing AoS states[16][8], this function operates
 * on __m512i soa_state[8] where each element packs 16 lanes of one
 * SHA256 state word.  This eliminates the costly load_state_word_16way /
 * store_16way scatter operations and allows the caller to keep data in
 * SIMD registers between pipeline stages.
 *
 * Parameters:
 *   soa_state : __m512i[8] — input/output, each holds 16 lanes of one state word
 *   blocks    : array of 16 pointers to 64-byte message blocks
 */
__attribute__((target("avx512f")))
void sha256_compress_avx512_soa(__m512i soa_state[8], const uint8_t *blocks[16])
{
    /* Load 16-lane initial state from SoA array */
    __m512i a = soa_state[0];
    __m512i b = soa_state[1];
    __m512i c = soa_state[2];
    __m512i d = soa_state[3];
    __m512i e = soa_state[4];
    __m512i f = soa_state[5];
    __m512i g = soa_state[6];
    __m512i h = soa_state[7];

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

    /* Accumulate initial state and write back to SoA array (no store_16way!) */
    soa_state[0] = _mm512_add_epi32(a, a0);
    soa_state[1] = _mm512_add_epi32(b, b0);
    soa_state[2] = _mm512_add_epi32(c, c0);
    soa_state[3] = _mm512_add_epi32(d, d0);
    soa_state[4] = _mm512_add_epi32(e, e0);
    soa_state[5] = _mm512_add_epi32(f, f0);
    soa_state[6] = _mm512_add_epi32(g, g0);
    soa_state[7] = _mm512_add_epi32(h, h0);
}

#endif /* __AVX512F__ */

