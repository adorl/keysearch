/*
 * sha256_avx2.c — SHA256 8-way AVX2并行压缩函数
 *
 * 同时处理8个独立消息块（每个块64字节），利用AVX2 256-bit寄存器
 * 将8个uint32_t打包为一个__m256i，每条SIMD指令同时推进8路状态
 *
 * 接口：
 *   void sha256_compress_avx2(uint32_t *states[8], const uint8_t *blocks[8])
 */

#ifdef __AVX2__

#include <immintrin.h>
#include <stdint.h>
#include "sha256.h"

/* 右旋转 */
#define V_ROR(x, n) _mm256_or_si256(_mm256_srli_epi32((x), (n)), _mm256_slli_epi32((x), 32 - (n)))

/* SHA256辅助函数（向量化版本） */
#define V_CH(x, y, z)   _mm256_xor_si256(_mm256_and_si256((x), (y)), _mm256_andnot_si256((x), (z)))
#define V_MAJ(x, y, z)  _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256((x), (y)), \
                            _mm256_and_si256((x), (z))), _mm256_and_si256((y), (z)))
#define V_EP0(x)        _mm256_xor_si256(_mm256_xor_si256(V_ROR((x), 2), V_ROR((x), 13)), V_ROR((x), 22))
#define V_EP1(x)        _mm256_xor_si256(_mm256_xor_si256(V_ROR((x), 6), V_ROR((x), 11)), V_ROR((x), 25))
#define V_SIG0(x)       _mm256_xor_si256(_mm256_xor_si256(V_ROR((x), 7), V_ROR((x), 18)), _mm256_srli_epi32((x), 3))
#define V_SIG1(x)       _mm256_xor_si256(_mm256_xor_si256(V_ROR((x), 17), V_ROR((x), 19)), _mm256_srli_epi32((x), 10))

/* 一轮SHA256（向量化）：同时推进8路状态 */
#define V_ROUND(a, b, c, d, e, f, g, h, k_val, w)                                           \
    do {                                                                                    \
        __m256i _k = _mm256_set1_epi32(k_val);                                              \
        __m256i _t1 = _mm256_add_epi32(                                                     \
            _mm256_add_epi32(_mm256_add_epi32((h), V_EP1(e)),                               \
            _mm256_add_epi32(V_CH(e, f, g), _k)), (w));                                     \
        __m256i _t2 = _mm256_add_epi32(V_EP0(a), V_MAJ(a, b, c));                           \
        (d) = _mm256_add_epi32((d), _t1);                                                   \
        (h) = _mm256_add_epi32(_t1, _t2);                                                   \
    } while (0)

/* 消息扩展 */
#define V_EXPAND(w0, w1, w9, w14) \
    _mm256_add_epi32(_mm256_add_epi32(V_SIG1(w14), (w9)), _mm256_add_epi32(V_SIG0(w1), (w0)))

/* 从8个block的第i个uint32_t（大端序）加载到__m256i */
static inline __m256i load_be32_8way(const uint8_t *const blocks[8], int i)
{
    /* 每个block中第i个字的偏移=i*4 */
    int off = i * 4;
    uint32_t v0 = ((uint32_t)blocks[0][off] << 24) | ((uint32_t)blocks[0][off + 1] << 16) | ((uint32_t)blocks[0][off + 2] << 8) | (uint32_t)blocks[0][off + 3];
    uint32_t v1 = ((uint32_t)blocks[1][off] << 24) | ((uint32_t)blocks[1][off + 1] << 16) | ((uint32_t)blocks[1][off + 2] << 8) | (uint32_t)blocks[1][off + 3];
    uint32_t v2 = ((uint32_t)blocks[2][off] << 24) | ((uint32_t)blocks[2][off + 1] << 16) | ((uint32_t)blocks[2][off + 2] << 8) | (uint32_t)blocks[2][off + 3];
    uint32_t v3 = ((uint32_t)blocks[3][off] << 24) | ((uint32_t)blocks[3][off + 1] << 16) | ((uint32_t)blocks[3][off + 2] << 8) | (uint32_t)blocks[3][off + 3];
    uint32_t v4 = ((uint32_t)blocks[4][off] << 24) | ((uint32_t)blocks[4][off + 1] << 16) | ((uint32_t)blocks[4][off + 2] << 8) | (uint32_t)blocks[4][off + 3];
    uint32_t v5 = ((uint32_t)blocks[5][off] << 24) | ((uint32_t)blocks[5][off + 1] << 16) | ((uint32_t)blocks[5][off + 2] << 8) | (uint32_t)blocks[5][off + 3];
    uint32_t v6 = ((uint32_t)blocks[6][off] << 24) | ((uint32_t)blocks[6][off + 1] << 16) | ((uint32_t)blocks[6][off + 2] << 8) | (uint32_t)blocks[6][off + 3];
    uint32_t v7 = ((uint32_t)blocks[7][off] << 24) | ((uint32_t)blocks[7][off + 1] << 16) | ((uint32_t)blocks[7][off + 2] << 8) | (uint32_t)blocks[7][off + 3];
    return _mm256_set_epi32((int)v7, (int)v6, (int)v5, (int)v4, (int)v3, (int)v2, (int)v1, (int)v0);
}

/* 将__m256i的8个lane分别写回8个state的第i个元素 */
static inline void store_8way(uint32_t *const states[8], int i, __m256i v)
{
    uint32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, v);
    states[0][i] = tmp[0];
    states[1][i] = tmp[1];
    states[2][i] = tmp[2];
    states[3][i] = tmp[3];
    states[4][i] = tmp[4];
    states[5][i] = tmp[5];
    states[6][i] = tmp[6];
    states[7][i] = tmp[7];
}

/*
 * 同时对8个独立的(state, block)对执行一次SHA256压缩
 */
__attribute__((target("avx2")))
void sha256_compress_avx2(uint32_t *states[8], const uint8_t *blocks[8])
{
    /* 加载 8 路初始状态 */
    __m256i a = _mm256_set_epi32((int)states[7][0], (int)states[6][0], (int)states[5][0], (int)states[4][0], (int)states[3][0], (int)states[2][0], (int)states[1][0], (int)states[0][0]);
    __m256i b = _mm256_set_epi32((int)states[7][1], (int)states[6][1], (int)states[5][1], (int)states[4][1], (int)states[3][1], (int)states[2][1], (int)states[1][1], (int)states[0][1]);
    __m256i c = _mm256_set_epi32((int)states[7][2], (int)states[6][2], (int)states[5][2], (int)states[4][2], (int)states[3][2], (int)states[2][2], (int)states[1][2], (int)states[0][2]);
    __m256i d = _mm256_set_epi32((int)states[7][3], (int)states[6][3], (int)states[5][3], (int)states[4][3], (int)states[3][3], (int)states[2][3], (int)states[1][3], (int)states[0][3]);
    __m256i e = _mm256_set_epi32((int)states[7][4], (int)states[6][4], (int)states[5][4], (int)states[4][4], (int)states[3][4], (int)states[2][4], (int)states[1][4], (int)states[0][4]);
    __m256i f = _mm256_set_epi32((int)states[7][5], (int)states[6][5], (int)states[5][5], (int)states[4][5], (int)states[3][5], (int)states[2][5], (int)states[1][5], (int)states[0][5]);
    __m256i g = _mm256_set_epi32((int)states[7][6], (int)states[6][6], (int)states[5][6], (int)states[4][6], (int)states[3][6], (int)states[2][6], (int)states[1][6], (int)states[0][6]);
    __m256i h = _mm256_set_epi32((int)states[7][7], (int)states[6][7], (int)states[5][7], (int)states[4][7], (int)states[3][7], (int)states[2][7], (int)states[1][7], (int)states[0][7]);

    /* 保存初始状态用于最后累加 */
    __m256i a0 = a, b0 = b, c0 = c, d0 = d, e0 = e, f0 = f, g0 = g, h0 = h;

    /* 加载 8 路消息字（大端序） */
    __m256i w0 = load_be32_8way(blocks, 0), w1 = load_be32_8way(blocks, 1);
    __m256i w2 = load_be32_8way(blocks, 2), w3 = load_be32_8way(blocks, 3);
    __m256i w4 = load_be32_8way(blocks, 4), w5 = load_be32_8way(blocks, 5);
    __m256i w6 = load_be32_8way(blocks, 6), w7 = load_be32_8way(blocks, 7);
    __m256i w8 = load_be32_8way(blocks, 8), w9 = load_be32_8way(blocks, 9);
    __m256i w10 = load_be32_8way(blocks, 10), w11 = load_be32_8way(blocks, 11);
    __m256i w12 = load_be32_8way(blocks, 12), w13 = load_be32_8way(blocks, 13);
    __m256i w14 = load_be32_8way(blocks, 14), w15 = load_be32_8way(blocks, 15);

    /* 轮0-15 */
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

    /* 轮16-31 */
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

    /* 轮32-47 */
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

    /* 轮48-63 */
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

    /* 累加回各路 state */
    a = _mm256_add_epi32(a, a0);
    b = _mm256_add_epi32(b, b0);
    c = _mm256_add_epi32(c, c0);
    d = _mm256_add_epi32(d, d0);
    e = _mm256_add_epi32(e, e0);
    f = _mm256_add_epi32(f, f0);
    g = _mm256_add_epi32(g, g0);
    h = _mm256_add_epi32(h, h0);

    /* 写回 8 路 state */
    store_8way(states, 0, a);
    store_8way(states, 1, b);
    store_8way(states, 2, c);
    store_8way(states, 3, d);
    store_8way(states, 4, e);
    store_8way(states, 5, f);
    store_8way(states, 6, g);
    store_8way(states, 7, h);
}

#endif /* __AVX2__ */

