/*
 * ripemd160_avx512.c — RIPEMD160 16-way AVX-512 parallel compression function (SoA variant)
 *
 * Processes 16 independent message blocks simultaneously,
 * using AVX-512 512-bit registers.
 *
 * SoA Interface:
 *   void ripemd160_compress_avx512_soa(__m512i soa_state[5], const __m512i w[16])
 */

#ifdef __AVX512F__

#include <immintrin.h>
#include <stdint.h>
#include "ripemd160.h"
#include "avx512_common.h"

/* Left rotation */
#define V_ROL(x, n) _mm512_or_si512(_mm512_slli_epi32((x), (n)), _mm512_srli_epi32((x), 32 - (n)))

/*
 * RIPEMD160 auxiliary functions (vectorized, single-instruction via AVX-512 ternarylogic)
 *
 * Each V_* compresses a 3-input boolean function into one vpternlogd instruction.
 * Truth table derivation: for each output bit, evaluate f(x,y,z) over all 8 input
 * combinations (x=bit7..0 of 0xF0, y=0xCC, z=0xAA), pack results into an 8-bit imm8.
 */
#define V_F(x, y, z)    _mm512_ternarylogic_epi32((x), (y), (z), 0x96) /* x ^ y ^ z */
#define V_G(x, y, z)    _mm512_ternarylogic_epi32((x), (y), (z), 0xCA) /* (x & y) | (~x & z) */
#define V_H(x, y, z)    _mm512_ternarylogic_epi32((x), (y), (z), 0x59) /* (x | ~y) ^ z */
#define V_I(x, y, z)    _mm512_ternarylogic_epi32((x), (y), (z), 0xE4) /* (x & z) | (y & ~z) */
#define V_J(x, y, z)    _mm512_ternarylogic_epi32((x), (y), (z), 0x2D) /* x ^ (y | ~z) */

/* Left chain step macros */
#define VRL_STEP(a, b, c, d, e, func_val, s)                                                \
    do {                                                                                    \
        __m512i _t = _mm512_add_epi32(V_ROL(_mm512_add_epi32((a), (func_val)), (s)), (e));  \
        (a) = (e);                                                                          \
        (e) = (d);                                                                          \
        (d) = V_ROL((c), 10);                                                               \
        (c) = (b);                                                                          \
        (b) = _t;                                                                           \
    } while (0)

/* Left chain round macros (func_val already includes x and k) */
#define VRL_F(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(V_F(b, c, d), (x)), s)
#define VRL_G(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_G(b, c, d), (x)), _mm512_set1_epi32(0x5A827999)), s)
#define VRL_H(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_H(b, c, d), (x)), _mm512_set1_epi32(0x6ED9EBA1)), s)
#define VRL_I(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_I(b, c, d), (x)), _mm512_set1_epi32((int)0x8F1BBCDC)), s)
#define VRL_J(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_J(b, c, d), (x)), _mm512_set1_epi32((int)0xA953FD4E)), s)

/* Right chain round macros */
#define VRR_J(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_J(b, c, d), (x)), _mm512_set1_epi32(0x50A28BE6)), s)
#define VRR_I(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_I(b, c, d), (x)), _mm512_set1_epi32(0x5C4DD124)), s)
#define VRR_H(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_H(b, c, d), (x)), _mm512_set1_epi32(0x6D703EF3)), s)
#define VRR_G(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_G(b, c, d), (x)), _mm512_set1_epi32(0x7A6D76E9)), s)
#define VRR_F(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(V_F(b, c, d), (x)), s)

/*
 * ripemd160_compress_avx512_soa — SoA variant with pre-loaded message words.
 *
 * Accepts __m512i soa_state[5] (input/output) and pre-loaded __m512i w[16]
 * message words. This eliminates all load_le32_contig/gather overhead and
 * rmd_store_16way scatter overhead, keeping data in SIMD registers throughout.
 *
 * Parameters:
 *   soa_state : __m512i[5] — input/output, each holds 16 lanes of one RIPEMD160 state word
 *   w         : const __m512i[16] — pre-loaded message words (little-endian)
 */
__attribute__((target("avx512f")))
void ripemd160_compress_avx512_soa(__m512i soa_state[5], const __m512i w[16])
{
    /* Load 16-lane initial state from SoA array */
    __m512i al = soa_state[0];
    __m512i bl = soa_state[1];
    __m512i cl = soa_state[2];
    __m512i dl = soa_state[3];
    __m512i el = soa_state[4];
    __m512i ar = al, br = bl, cr = cl, dr = dl, er = el;

    /* Save initial state */
    __m512i s0 = al, s1 = bl, s2 = cl, s3 = dl, s4 = el;

    /* Use pre-loaded message words directly */
    __m512i w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    __m512i w4 = w[4], w5 = w[5], w6 = w[6], w7 = w[7];
    __m512i w8 = w[8], w9 = w[9], w10 = w[10], w11 = w[11];
    __m512i w12 = w[12], w13 = w[13], w14 = w[14], w15 = w[15];

    /* Left chain: rounds 0-15, F */
    VRL_F(al, bl, cl, dl, el, w0, 11);
    VRL_F(al, bl, cl, dl, el, w1, 14);
    VRL_F(al, bl, cl, dl, el, w2, 15);
    VRL_F(al, bl, cl, dl, el, w3, 12);
    VRL_F(al, bl, cl, dl, el, w4, 5);
    VRL_F(al, bl, cl, dl, el, w5, 8);
    VRL_F(al, bl, cl, dl, el, w6, 7);
    VRL_F(al, bl, cl, dl, el, w7, 9);
    VRL_F(al, bl, cl, dl, el, w8, 11);
    VRL_F(al, bl, cl, dl, el, w9, 13);
    VRL_F(al, bl, cl, dl, el, w10, 14);
    VRL_F(al, bl, cl, dl, el, w11, 15);
    VRL_F(al, bl, cl, dl, el, w12, 6);
    VRL_F(al, bl, cl, dl, el, w13, 7);
    VRL_F(al, bl, cl, dl, el, w14, 9);
    VRL_F(al, bl, cl, dl, el, w15, 8);
    /* Left chain: rounds 16-31, G */
    VRL_G(al, bl, cl, dl, el, w7, 7);
    VRL_G(al, bl, cl, dl, el, w4, 6);
    VRL_G(al, bl, cl, dl, el, w13, 8);
    VRL_G(al, bl, cl, dl, el, w1, 13);
    VRL_G(al, bl, cl, dl, el, w10, 11);
    VRL_G(al, bl, cl, dl, el, w6, 9);
    VRL_G(al, bl, cl, dl, el, w15, 7);
    VRL_G(al, bl, cl, dl, el, w3, 15);
    VRL_G(al, bl, cl, dl, el, w12, 7);
    VRL_G(al, bl, cl, dl, el, w0, 12);
    VRL_G(al, bl, cl, dl, el, w9, 15);
    VRL_G(al, bl, cl, dl, el, w5, 9);
    VRL_G(al, bl, cl, dl, el, w2, 11);
    VRL_G(al, bl, cl, dl, el, w14, 7);
    VRL_G(al, bl, cl, dl, el, w11, 13);
    VRL_G(al, bl, cl, dl, el, w8, 12);
    /* Left chain: rounds 32-47, H */
    VRL_H(al, bl, cl, dl, el, w3, 11);
    VRL_H(al, bl, cl, dl, el, w10, 13);
    VRL_H(al, bl, cl, dl, el, w14, 6);
    VRL_H(al, bl, cl, dl, el, w4, 7);
    VRL_H(al, bl, cl, dl, el, w9, 14);
    VRL_H(al, bl, cl, dl, el, w15, 9);
    VRL_H(al, bl, cl, dl, el, w8, 13);
    VRL_H(al, bl, cl, dl, el, w1, 15);
    VRL_H(al, bl, cl, dl, el, w2, 14);
    VRL_H(al, bl, cl, dl, el, w7, 8);
    VRL_H(al, bl, cl, dl, el, w0, 13);
    VRL_H(al, bl, cl, dl, el, w6, 6);
    VRL_H(al, bl, cl, dl, el, w13, 5);
    VRL_H(al, bl, cl, dl, el, w11, 12);
    VRL_H(al, bl, cl, dl, el, w5, 7);
    VRL_H(al, bl, cl, dl, el, w12, 5);
    /* Left chain: rounds 48-63, I */
    VRL_I(al, bl, cl, dl, el, w1, 11);
    VRL_I(al, bl, cl, dl, el, w9, 12);
    VRL_I(al, bl, cl, dl, el, w11, 14);
    VRL_I(al, bl, cl, dl, el, w10, 15);
    VRL_I(al, bl, cl, dl, el, w0, 14);
    VRL_I(al, bl, cl, dl, el, w8, 15);
    VRL_I(al, bl, cl, dl, el, w12, 9);
    VRL_I(al, bl, cl, dl, el, w4, 8);
    VRL_I(al, bl, cl, dl, el, w13, 9);
    VRL_I(al, bl, cl, dl, el, w3, 14);
    VRL_I(al, bl, cl, dl, el, w7, 5);
    VRL_I(al, bl, cl, dl, el, w15, 6);
    VRL_I(al, bl, cl, dl, el, w14, 8);
    VRL_I(al, bl, cl, dl, el, w5, 6);
    VRL_I(al, bl, cl, dl, el, w6, 5);
    VRL_I(al, bl, cl, dl, el, w2, 12);
    /* Left chain: rounds 64-79, J */
    VRL_J(al, bl, cl, dl, el, w4, 9);
    VRL_J(al, bl, cl, dl, el, w0, 15);
    VRL_J(al, bl, cl, dl, el, w5, 5);
    VRL_J(al, bl, cl, dl, el, w9, 11);
    VRL_J(al, bl, cl, dl, el, w7, 6);
    VRL_J(al, bl, cl, dl, el, w12, 8);
    VRL_J(al, bl, cl, dl, el, w2, 13);
    VRL_J(al, bl, cl, dl, el, w10, 12);
    VRL_J(al, bl, cl, dl, el, w14, 5);
    VRL_J(al, bl, cl, dl, el, w1, 12);
    VRL_J(al, bl, cl, dl, el, w3, 13);
    VRL_J(al, bl, cl, dl, el, w8, 14);
    VRL_J(al, bl, cl, dl, el, w11, 11);
    VRL_J(al, bl, cl, dl, el, w6, 8);
    VRL_J(al, bl, cl, dl, el, w15, 5);
    VRL_J(al, bl, cl, dl, el, w13, 6);

    /* Right chain: rounds 0-15, J */
    VRR_J(ar, br, cr, dr, er, w5, 8);
    VRR_J(ar, br, cr, dr, er, w14, 9);
    VRR_J(ar, br, cr, dr, er, w7, 9);
    VRR_J(ar, br, cr, dr, er, w0, 11);
    VRR_J(ar, br, cr, dr, er, w9, 13);
    VRR_J(ar, br, cr, dr, er, w2, 15);
    VRR_J(ar, br, cr, dr, er, w11, 15);
    VRR_J(ar, br, cr, dr, er, w4, 5);
    VRR_J(ar, br, cr, dr, er, w13, 7);
    VRR_J(ar, br, cr, dr, er, w6, 7);
    VRR_J(ar, br, cr, dr, er, w15, 8);
    VRR_J(ar, br, cr, dr, er, w8, 11);
    VRR_J(ar, br, cr, dr, er, w1, 14);
    VRR_J(ar, br, cr, dr, er, w10, 14);
    VRR_J(ar, br, cr, dr, er, w3, 12);
    VRR_J(ar, br, cr, dr, er, w12, 6);
    /* Right chain: rounds 16-31, I */
    VRR_I(ar, br, cr, dr, er, w6, 9);
    VRR_I(ar, br, cr, dr, er, w11, 13);
    VRR_I(ar, br, cr, dr, er, w3, 15);
    VRR_I(ar, br, cr, dr, er, w7, 7);
    VRR_I(ar, br, cr, dr, er, w0, 12);
    VRR_I(ar, br, cr, dr, er, w13, 8);
    VRR_I(ar, br, cr, dr, er, w5, 9);
    VRR_I(ar, br, cr, dr, er, w10, 11);
    VRR_I(ar, br, cr, dr, er, w14, 7);
    VRR_I(ar, br, cr, dr, er, w15, 7);
    VRR_I(ar, br, cr, dr, er, w8, 12);
    VRR_I(ar, br, cr, dr, er, w12, 7);
    VRR_I(ar, br, cr, dr, er, w4, 6);
    VRR_I(ar, br, cr, dr, er, w9, 15);
    VRR_I(ar, br, cr, dr, er, w1, 13);
    VRR_I(ar, br, cr, dr, er, w2, 11);
    /* Right chain: rounds 32-47, H */
    VRR_H(ar, br, cr, dr, er, w15, 9);
    VRR_H(ar, br, cr, dr, er, w5, 7);
    VRR_H(ar, br, cr, dr, er, w1, 15);
    VRR_H(ar, br, cr, dr, er, w3, 11);
    VRR_H(ar, br, cr, dr, er, w7, 8);
    VRR_H(ar, br, cr, dr, er, w14, 6);
    VRR_H(ar, br, cr, dr, er, w6, 6);
    VRR_H(ar, br, cr, dr, er, w9, 14);
    VRR_H(ar, br, cr, dr, er, w11, 12);
    VRR_H(ar, br, cr, dr, er, w8, 13);
    VRR_H(ar, br, cr, dr, er, w12, 5);
    VRR_H(ar, br, cr, dr, er, w2, 14);
    VRR_H(ar, br, cr, dr, er, w10, 13);
    VRR_H(ar, br, cr, dr, er, w0, 13);
    VRR_H(ar, br, cr, dr, er, w4, 7);
    VRR_H(ar, br, cr, dr, er, w13, 5);
    /* Right chain: rounds 48-63, G */
    VRR_G(ar, br, cr, dr, er, w8, 15);
    VRR_G(ar, br, cr, dr, er, w6, 5);
    VRR_G(ar, br, cr, dr, er, w4, 8);
    VRR_G(ar, br, cr, dr, er, w1, 11);
    VRR_G(ar, br, cr, dr, er, w3, 14);
    VRR_G(ar, br, cr, dr, er, w11, 14);
    VRR_G(ar, br, cr, dr, er, w15, 6);
    VRR_G(ar, br, cr, dr, er, w0, 14);
    VRR_G(ar, br, cr, dr, er, w5, 6);
    VRR_G(ar, br, cr, dr, er, w12, 9);
    VRR_G(ar, br, cr, dr, er, w2, 12);
    VRR_G(ar, br, cr, dr, er, w13, 9);
    VRR_G(ar, br, cr, dr, er, w9, 12);
    VRR_G(ar, br, cr, dr, er, w7, 5);
    VRR_G(ar, br, cr, dr, er, w10, 15);
    VRR_G(ar, br, cr, dr, er, w14, 8);
    /* Right chain: rounds 64-79, F */
    VRR_F(ar, br, cr, dr, er, w12, 8);
    VRR_F(ar, br, cr, dr, er, w15, 5);
    VRR_F(ar, br, cr, dr, er, w10, 12);
    VRR_F(ar, br, cr, dr, er, w4, 9);
    VRR_F(ar, br, cr, dr, er, w1, 12);
    VRR_F(ar, br, cr, dr, er, w5, 5);
    VRR_F(ar, br, cr, dr, er, w8, 14);
    VRR_F(ar, br, cr, dr, er, w7, 6);
    VRR_F(ar, br, cr, dr, er, w6, 8);
    VRR_F(ar, br, cr, dr, er, w2, 13);
    VRR_F(ar, br, cr, dr, er, w13, 6);
    VRR_F(ar, br, cr, dr, er, w14, 5);
    VRR_F(ar, br, cr, dr, er, w0, 15);
    VRR_F(ar, br, cr, dr, er, w3, 13);
    VRR_F(ar, br, cr, dr, er, w9, 11);
    VRR_F(ar, br, cr, dr, er, w11, 11);

    /* Merge left and right chains, write back to SoA state (no rmd_store_16way!) */
    soa_state[0] = _mm512_add_epi32(_mm512_add_epi32(s1, cl), dr);
    soa_state[1] = _mm512_add_epi32(_mm512_add_epi32(s2, dl), er);
    soa_state[2] = _mm512_add_epi32(_mm512_add_epi32(s3, el), ar);
    soa_state[3] = _mm512_add_epi32(_mm512_add_epi32(s4, al), br);
    soa_state[4] = _mm512_add_epi32(_mm512_add_epi32(s0, bl), cr);
}

#endif /* __AVX512F__ */

