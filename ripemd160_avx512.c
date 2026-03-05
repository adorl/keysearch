/*
 * ripemd160_avx512.c — RIPEMD160 16-way AVX-512并行压缩函数
 *
 * 同时处理16个独立消息块（每个块64字节），利用 AVX-512 512-bit寄存器
 *
 * 接口：
 *   void ripemd160_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16])
 */

#ifdef __AVX512F__

#include <immintrin.h>
#include <stdint.h>
#include "ripemd160.h"

/* 左旋转 */
#define V_ROL(x, n) _mm512_or_si512(_mm512_slli_epi32((x), (n)), _mm512_srli_epi32((x), 32 - (n)))

/* RIPEMD160辅助函数（向量化版本） */
#define V_F(x, y, z)    _mm512_xor_si512(_mm512_xor_si512((x), (y)), (z))
#define V_G(x, y, z)    _mm512_or_si512(_mm512_and_si512((x), (y)), _mm512_andnot_si512((x), (z)))
#define V_H(x, y, z)    _mm512_xor_si512(_mm512_or_si512((x), _mm512_xor_si512(_mm512_set1_epi32(-1), (y))), (z))
#define V_I(x, y, z)    _mm512_or_si512(_mm512_and_si512((x), (z)), _mm512_andnot_si512((z), (y)))
#define V_J(x, y, z)    _mm512_xor_si512((x), _mm512_or_si512((y), _mm512_xor_si512(_mm512_set1_epi32(-1), (z))))

/* 左链步骤宏 */
#define VRL_STEP(a, b, c, d, e, func_val, s)                                                \
    do {                                                                                    \
        __m512i _t = _mm512_add_epi32(V_ROL(_mm512_add_epi32((a), (func_val)), (s)), (e));  \
        (a) = (e);                                                                          \
        (e) = (d);                                                                          \
        (d) = V_ROL((c), 10);                                                               \
        (c) = (b);                                                                          \
        (b) = _t;                                                                           \
    } while (0)

/* 左链各轮宏（func_val 已包含 x 和 k） */
#define VRL_F(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(V_F(b, c, d), (x)), s)
#define VRL_G(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_G(b, c, d), (x)), _mm512_set1_epi32(0x5A827999)), s)
#define VRL_H(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_H(b, c, d), (x)), _mm512_set1_epi32(0x6ED9EBA1)), s)
#define VRL_I(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_I(b, c, d), (x)), _mm512_set1_epi32((int)0x8F1BBCDC)), s)
#define VRL_J(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_J(b, c, d), (x)), _mm512_set1_epi32((int)0xA953FD4E)), s)

/* 右链各轮宏 */
#define VRR_J(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_J(b, c, d), (x)), _mm512_set1_epi32(0x50A28BE6)), s)
#define VRR_I(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_I(b, c, d), (x)), _mm512_set1_epi32(0x5C4DD124)), s)
#define VRR_H(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_H(b, c, d), (x)), _mm512_set1_epi32(0x6D703EF3)), s)
#define VRR_G(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(_mm512_add_epi32(V_G(b, c, d), (x)), _mm512_set1_epi32(0x7A6D76E9)), s)
#define VRR_F(a, b, c, d, e, x, s)  VRL_STEP(a, b, c, d, e, _mm512_add_epi32(V_F(b, c, d), (x)), s)

/* 从16个block的第i个uint32_t（小端序）加载到__m512i */
static inline __m512i load_le32_16way(const uint8_t *const blocks[16], int i)
{
    int off = i * 4;
    uint32_t v0 = (uint32_t)blocks[0][off] | ((uint32_t)blocks[0][off + 1] << 8) | ((uint32_t)blocks[0][off + 2] << 16) | ((uint32_t)blocks[0][off + 3] << 24);
    uint32_t v1 = (uint32_t)blocks[1][off] | ((uint32_t)blocks[1][off + 1] << 8) | ((uint32_t)blocks[1][off + 2] << 16) | ((uint32_t)blocks[1][off + 3] << 24);
    uint32_t v2 = (uint32_t)blocks[2][off] | ((uint32_t)blocks[2][off + 1] << 8) | ((uint32_t)blocks[2][off + 2] << 16) | ((uint32_t)blocks[2][off + 3] << 24);
    uint32_t v3 = (uint32_t)blocks[3][off] | ((uint32_t)blocks[3][off + 1] << 8) | ((uint32_t)blocks[3][off + 2] << 16) | ((uint32_t)blocks[3][off + 3] << 24);
    uint32_t v4 = (uint32_t)blocks[4][off] | ((uint32_t)blocks[4][off + 1] << 8) | ((uint32_t)blocks[4][off + 2] << 16) | ((uint32_t)blocks[4][off + 3] << 24);
    uint32_t v5 = (uint32_t)blocks[5][off] | ((uint32_t)blocks[5][off + 1] << 8) | ((uint32_t)blocks[5][off + 2] << 16) | ((uint32_t)blocks[5][off + 3] << 24);
    uint32_t v6 = (uint32_t)blocks[6][off] | ((uint32_t)blocks[6][off + 1] << 8) | ((uint32_t)blocks[6][off + 2] << 16) | ((uint32_t)blocks[6][off + 3] << 24);
    uint32_t v7 = (uint32_t)blocks[7][off] | ((uint32_t)blocks[7][off + 1] << 8) | ((uint32_t)blocks[7][off + 2] << 16) | ((uint32_t)blocks[7][off + 3] << 24);
    uint32_t v8 = (uint32_t)blocks[8][off] | ((uint32_t)blocks[8][off + 1] << 8) | ((uint32_t)blocks[8][off + 2] << 16) | ((uint32_t)blocks[8][off + 3] << 24);
    uint32_t v9 = (uint32_t)blocks[9][off] | ((uint32_t)blocks[9][off + 1] << 8) | ((uint32_t)blocks[9][off + 2] << 16) | ((uint32_t)blocks[9][off + 3] << 24);
    uint32_t v10 = (uint32_t)blocks[10][off] | ((uint32_t)blocks[10][off + 1] << 8) | ((uint32_t)blocks[10][off + 2] << 16) | ((uint32_t)blocks[10][off + 3] << 24);
    uint32_t v11 = (uint32_t)blocks[11][off] | ((uint32_t)blocks[11][off + 1] << 8) | ((uint32_t)blocks[11][off + 2] << 16) | ((uint32_t)blocks[11][off + 3] << 24);
    uint32_t v12 = (uint32_t)blocks[12][off] | ((uint32_t)blocks[12][off + 1] << 8) | ((uint32_t)blocks[12][off + 2] << 16) | ((uint32_t)blocks[12][off + 3] << 24);
    uint32_t v13 = (uint32_t)blocks[13][off] | ((uint32_t)blocks[13][off + 1] << 8) | ((uint32_t)blocks[13][off + 2] << 16) | ((uint32_t)blocks[13][off + 3] << 24);
    uint32_t v14 = (uint32_t)blocks[14][off] | ((uint32_t)blocks[14][off + 1] << 8) | ((uint32_t)blocks[14][off + 2] << 16) | ((uint32_t)blocks[14][off + 3] << 24);
    uint32_t v15 = (uint32_t)blocks[15][off] | ((uint32_t)blocks[15][off + 1] << 8) | ((uint32_t)blocks[15][off + 2] << 16) | ((uint32_t)blocks[15][off + 3] << 24);
    return _mm512_set_epi32((int)v15, (int)v14, (int)v13, (int)v12, (int)v11, (int)v10, (int)v9, (int)v8,
                            (int)v7, (int)v6, (int)v5, (int)v4, (int)v3, (int)v2, (int)v1, (int)v0);
}

/* 将__m512i的16个lane分别写回16个state的第i个元素 */
static inline void rmd_store_16way(uint32_t *const states[16], int i, __m512i v)
{
    uint32_t tmp[16];
    _mm512_storeu_si512((__m512i *)tmp, v);
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
 * ripemd160_compress_avx512 — 16-way并行RIPEMD160压缩
 *
 * 同时对16个独立的(state, block)对执行一次RIPEMD160压缩
 */
__attribute__((target("avx512f")))
void ripemd160_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16])
{
    /* 加载16路初始状态 */
    __m512i al = _mm512_set_epi32((int)states[15][0], (int)states[14][0], (int)states[13][0], (int)states[12][0],
                                  (int)states[11][0], (int)states[10][0], (int)states[9][0], (int)states[8][0],
                                  (int)states[7][0], (int)states[6][0], (int)states[5][0], (int)states[4][0],
                                  (int)states[3][0], (int)states[2][0], (int)states[1][0], (int)states[0][0]);
    __m512i bl = _mm512_set_epi32((int)states[15][1], (int)states[14][1], (int)states[13][1], (int)states[12][1],
                                  (int)states[11][1], (int)states[10][1], (int)states[9][1], (int)states[8][1],
                                  (int)states[7][1], (int)states[6][1], (int)states[5][1], (int)states[4][1],
                                  (int)states[3][1], (int)states[2][1], (int)states[1][1], (int)states[0][1]);
    __m512i cl = _mm512_set_epi32((int)states[15][2], (int)states[14][2], (int)states[13][2], (int)states[12][2],
                                  (int)states[11][2], (int)states[10][2], (int)states[9][2], (int)states[8][2],
                                  (int)states[7][2], (int)states[6][2], (int)states[5][2], (int)states[4][2],
                                  (int)states[3][2], (int)states[2][2], (int)states[1][2], (int)states[0][2]);
    __m512i dl = _mm512_set_epi32((int)states[15][3], (int)states[14][3], (int)states[13][3], (int)states[12][3],
                                  (int)states[11][3], (int)states[10][3], (int)states[9][3], (int)states[8][3],
                                  (int)states[7][3], (int)states[6][3], (int)states[5][3], (int)states[4][3],
                                  (int)states[3][3], (int)states[2][3], (int)states[1][3], (int)states[0][3]);
    __m512i el = _mm512_set_epi32((int)states[15][4], (int)states[14][4], (int)states[13][4], (int)states[12][4],
                                  (int)states[11][4], (int)states[10][4], (int)states[9][4], (int)states[8][4],
                                  (int)states[7][4], (int)states[6][4], (int)states[5][4], (int)states[4][4],
                                  (int)states[3][4], (int)states[2][4], (int)states[1][4], (int)states[0][4]);
    __m512i ar = al, br = bl, cr = cl, dr = dl, er = el;

    /* 保存初始状态 */
    __m512i s0 = al, s1 = bl, s2 = cl, s3 = dl, s4 = el;

    /* 加载16路消息字（小端序） */
    __m512i w0 = load_le32_16way(blocks, 0), w1 = load_le32_16way(blocks, 1);
    __m512i w2 = load_le32_16way(blocks, 2), w3 = load_le32_16way(blocks, 3);
    __m512i w4 = load_le32_16way(blocks, 4), w5 = load_le32_16way(blocks, 5);
    __m512i w6 = load_le32_16way(blocks, 6), w7 = load_le32_16way(blocks, 7);
    __m512i w8 = load_le32_16way(blocks, 8), w9 = load_le32_16way(blocks, 9);
    __m512i w10 = load_le32_16way(blocks, 10), w11 = load_le32_16way(blocks, 11);
    __m512i w12 = load_le32_16way(blocks, 12), w13 = load_le32_16way(blocks, 13);
    __m512i w14 = load_le32_16way(blocks, 14), w15 = load_le32_16way(blocks, 15);

    /* 左链：轮0-15，F */
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
    /* 左链：轮16-31，G */
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
    /* 左链：轮32-47，H */
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
    /* 左链：轮48-63，I */
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
    /* 左链：轮64-79，J */
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

    /* 右链：轮0-15，J */
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
    /* 右链：轮16-31，I */
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
    /* 右链：轮32-47，H */
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
    /* 右链：轮48-63，G */
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
    /* 右链：轮64-79，F */
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

    /* 合并左右链，更新状态
     * RIPEMD160 合并顺序：
     *   new[0] = h1 + cl + dr
     *   new[1] = h2 + dl + er
     *   new[2] = h3 + el + ar
     *   new[3] = h4 + al + br
     *   new[4] = h0 + bl + cr
     */
    __m512i new0 = _mm512_add_epi32(_mm512_add_epi32(s1, cl), dr);
    __m512i new1 = _mm512_add_epi32(_mm512_add_epi32(s2, dl), er);
    __m512i new2 = _mm512_add_epi32(_mm512_add_epi32(s3, el), ar);
    __m512i new3 = _mm512_add_epi32(_mm512_add_epi32(s4, al), br);
    __m512i new4 = _mm512_add_epi32(_mm512_add_epi32(s0, bl), cr);

    rmd_store_16way(states, 0, new0);
    rmd_store_16way(states, 1, new1);
    rmd_store_16way(states, 2, new2);
    rmd_store_16way(states, 3, new3);
    rmd_store_16way(states, 4, new4);
}

#endif /* __AVX512F__ */

