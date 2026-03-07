/*
 * secp256k1_keygen_avx512.c
 *
 * AVX-512 IFMA 16路并行有限域运算与点加法实现。
 *
 * 布局：SoA（Structure of Arrays）
 *   secp256k1_fe_16x.n[i] 是一个 __m512i，其8个64-bit lane分别存储
 *   16个点中第i个limb（每个lane存2个点，共8个lane×2=16个点）。
 *
 * 实际设计（本实现采用）：
 *   n[i] 为单个 __m512i，8个lane存16个点中的8个点（低8路）
 *   另一个 __m512i 存高8路，共用 n[i][0..1] 两个 __m512i。
 *   共 5×2=10 个 __m512i。
 *
 * 编译要求：-mavx512f -mavx512ifma
 */

#ifdef __AVX512IFMA__

#include "secp256k1_keygen.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/*
 * secp256k1_fe_16x：16路并行有限域元素（SoA布局）
 *
 * n[i][0]: 点0..7  的第i个limb（8个64-bit lane）
 * n[i][1]: 点8..15 的第i个limb（8个64-bit lane）
 *
 * 共5个limb × 2个__m512i = 10个__m512i寄存器
 */
typedef struct {
    __m512i n[5][2];
} secp256k1_fe_16x;

/*
 * fe_16x_load：将16个 secp256k1_fe 转换为 SoA 布局
 */
static void fe_16x_load(secp256k1_fe_16x *dst, const secp256k1_fe src[16])
{
    for (int i = 0; i < 5; i++) {
        dst->n[i][0] = _mm512_set_epi64(
            (int64_t)src[7].n[i],
            (int64_t)src[6].n[i],
            (int64_t)src[5].n[i],
            (int64_t)src[4].n[i],
            (int64_t)src[3].n[i],
            (int64_t)src[2].n[i],
            (int64_t)src[1].n[i],
            (int64_t)src[0].n[i]
        );
        dst->n[i][1] = _mm512_set_epi64(
            (int64_t)src[15].n[i],
            (int64_t)src[14].n[i],
            (int64_t)src[13].n[i],
            (int64_t)src[12].n[i],
            (int64_t)src[11].n[i],
            (int64_t)src[10].n[i],
            (int64_t)src[9].n[i],
            (int64_t)src[8].n[i]
        );
    }
}

/*
 * fe_16x_store：将 SoA 布局转换回16个 secp256k1_fe
 */
static void fe_16x_store(secp256k1_fe dst[16], const secp256k1_fe_16x *src)
{
    uint64_t lo[5][8];
    uint64_t hi[5][8];
    for (int i = 0; i < 5; i++) {
        _mm512_storeu_si512((__m512i *)lo[i], src->n[i][0]);
        _mm512_storeu_si512((__m512i *)hi[i], src->n[i][1]);
    }
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 5; i++) {
            dst[j].n[i]     = lo[i][j];
            dst[j + 8].n[i] = hi[i][j];
        }
    }
}

/*
 * 基本域运算
 */

/*
 * fe_16x_add：16路并行域加法（不含模归约，magnitude累加）
 */
static void fe_16x_add(secp256k1_fe_16x *r,
                       const secp256k1_fe_16x *a,
                       const secp256k1_fe_16x *b)
{
    for (int i = 0; i < 5; i++) {
        r->n[i][0] = _mm512_add_epi64(a->n[i][0], b->n[i][0]);
        r->n[i][1] = _mm512_add_epi64(a->n[i][1], b->n[i][1]);
    }
}

/*
 * fe_16x_negate：16路并行域取反
 * r = -a，输入magnitude为m，输出magnitude为m+1
 */
static void fe_16x_negate(secp256k1_fe_16x *r,
                          const secp256k1_fe_16x *a,
                          int m)
{
    uint64_t c0 = 0xFFFFEFFFFFC2FULL * 2 * (uint64_t)(m + 1);
    uint64_t c1 = 0xFFFFFFFFFFFFFULL * 2 * (uint64_t)(m + 1);
    uint64_t c4 = 0x0FFFFFFFFFFFFULL * 2 * (uint64_t)(m + 1);

    __m512i v0 = _mm512_set1_epi64((int64_t)c0);
    __m512i v1 = _mm512_set1_epi64((int64_t)c1);
    __m512i v4 = _mm512_set1_epi64((int64_t)c4);

    r->n[0][0] = _mm512_sub_epi64(v0, a->n[0][0]);
    r->n[0][1] = _mm512_sub_epi64(v0, a->n[0][1]);
    for (int i = 1; i <= 3; i++) {
        r->n[i][0] = _mm512_sub_epi64(v1, a->n[i][0]);
        r->n[i][1] = _mm512_sub_epi64(v1, a->n[i][1]);
    }
    r->n[4][0] = _mm512_sub_epi64(v4, a->n[4][0]);
    r->n[4][1] = _mm512_sub_epi64(v4, a->n[4][1]);
}

/*
 * fe_16x_normalize_weak：16路并行弱归约
 * 将 magnitude降至1
 */
static void fe_16x_normalize_weak(secp256k1_fe_16x *r)
{
    __m512i mask52 = _mm512_set1_epi64((int64_t)0xFFFFFFFFFFFFFULL);
    __m512i mask48 = _mm512_set1_epi64((int64_t)0x0FFFFFFFFFFFFULL);

    for (int half = 0; half < 2; half++) {
        __m512i t0 = r->n[0][half], t1 = r->n[1][half],
                t2 = r->n[2][half], t3 = r->n[3][half], t4 = r->n[4][half];

        /* x = t4 >> 48; t4 &= 0x0FFFFFFFFFFFF */
        __m512i x = _mm512_srli_epi64(t4, 48);
        t4 = _mm512_and_si512(t4, mask48);

        /* t0 += x * 0x1000003D1
         * x 最多 4 位，0x1000003D1 = (1<<32) + 0x3D1
         * = (x << 32) + x * 0x3D1
         * 用 _mm512_mul_epu32 计算低32位乘积（x <= 0xF，0x3D1 <= 0xFFF，积 <= 16位，安全） */
        __m512i R1_lo = _mm512_set1_epi64(0x3D1LL);
        __m512i xR = _mm512_add_epi64(
            _mm512_slli_epi64(x, 32),
            _mm512_mul_epu32(x, R1_lo)
        );
        t0 = _mm512_add_epi64(t0, xR);

        /* 进位传播 */
        __m512i c;
        c  = _mm512_srli_epi64(t0, 52); t0 = _mm512_and_si512(t0, mask52);
        t1 = _mm512_add_epi64(t1, c);
        c  = _mm512_srli_epi64(t1, 52); t1 = _mm512_and_si512(t1, mask52);
        t2 = _mm512_add_epi64(t2, c);
        c  = _mm512_srli_epi64(t2, 52); t2 = _mm512_and_si512(t2, mask52);
        t3 = _mm512_add_epi64(t3, c);
        c  = _mm512_srli_epi64(t3, 52); t3 = _mm512_and_si512(t3, mask52);
        t4 = _mm512_add_epi64(t4, c);

        r->n[0][half] = t0; r->n[1][half] = t1; r->n[2][half] = t2;
        r->n[3][half] = t3; r->n[4][half] = t4;
    }
}

/*
 * AVX-512 IFMA 域乘法
 */

/*
 * 128-bit 向量累加器：(hi, lo) 表示 hi*2^64 + lo
 * 使用 __m512i 对，每个 lane 独立维护一个 128-bit 累加器
 */
typedef struct {
    __m512i lo;
    __m512i hi;
} u128_16x;

static inline u128_16x u128_16x_zero(void)
{
    u128_16x r;
    r.lo = _mm512_setzero_si512();
    r.hi = _mm512_setzero_si512();
    return r;
}

/*
 * acc += a * b（使用 AVX-512 IFMA 指令：VPMADD52LUQ / VPMADD52HUQ）
 * a, b 均为52-bit limb（存于64-bit lane）
 * VPMADD52LUQ: acc.lo += (a * b) & ((1<<52)-1)  （低52位）
 * VPMADD52HUQ: acc.hi += (a * b) >> 52           （高52位）
 */
static inline u128_16x u128_16x_accum_mul(u128_16x acc, __m512i a, __m512i b)
{
    u128_16x r;
    r.lo = _mm512_madd52lo_epu64(acc.lo, a, b);
    r.hi = _mm512_madd52hi_epu64(acc.hi, a, b);
    return r;
}

/*
 * acc += scalar * b（scalar 为广播的64-bit常量）
 */
static inline u128_16x u128_16x_accum_mul_scalar(u128_16x acc, uint64_t scalar, __m512i b)
{
    __m512i s = _mm512_set1_epi64((int64_t)scalar);
    return u128_16x_accum_mul(acc, s, b);
}

/*
 * 取 acc 的低52位，返回低52位值，acc >>= 52
 *
 * acc 真实值 = acc.lo + acc.hi * 2^52
 * 低52位 = acc.lo & M
 * 右移52位后，新真实值 = (acc.lo >> 52) + acc.hi
 * 因此：
 *   新 acc.lo = acc.hi + (acc.lo >> 52)
 *   新 acc.hi = 0
 *
 * 注意：acc.hi 存储的是 acc >> 52 的部分，右移 52 位后
 * acc.hi 直接成为新的低位，加上 acc.lo 的溢出部分（acc.lo >> 52，最多 12 位）。
 */
static inline __m512i u128_16x_extract52(u128_16x *acc)
{
    __m512i mask52 = _mm512_set1_epi64((int64_t)0xFFFFFFFFFFFFFULL);
    __m512i r = _mm512_and_si512(acc->lo, mask52);
    /* acc >>= 52: 新真实值 = acc.hi + (acc.lo >> 52) */
    __m512i lo_carry = _mm512_srli_epi64(acc->lo, 52);  /* acc.lo >> 52，最多 12 位 */
    acc->lo = _mm512_add_epi64(acc->hi, lo_carry);
    acc->hi = _mm512_setzero_si512();
    return r;
}

/*
 * 取 acc 的低64位，acc >>= 64
 *
 * 真实值 = acc.lo + acc.hi * 2^52
 * 低64位 = acc.lo + (acc.hi & 0xFFF) * 2^52  （mod 2^64，64位截断）
 * 右移64位后 = acc.hi >> 12 + carry
 * 其中 carry = 1 若加法溢出 64 位，否则 0
 */
static inline __m512i u128_16x_extract64(u128_16x *acc)
{
    __m512i mask12 = _mm512_set1_epi64(0xFFFLL);
    __m512i acc_hi_lo12 = _mm512_and_si512(acc->hi, mask12);
    __m512i acc_hi_hi   = _mm512_srli_epi64(acc->hi, 12);

    /* r = acc.lo + (acc.hi & 0xFFF) * 2^52（低64位，自动截断） */
    __m512i hi_contrib = _mm512_slli_epi64(acc_hi_lo12, 52);
    __m512i r = _mm512_add_epi64(acc->lo, hi_contrib);

    /* carry = (r < acc.lo)，即加法是否溢出 */
    __mmask8 carry_mask = _mm512_cmplt_epu64_mask(r, acc->lo);
    __m512i carry = _mm512_maskz_set1_epi64(carry_mask, 1);

    acc->lo = _mm512_add_epi64(acc_hi_hi, carry);
    acc->hi = _mm512_setzero_si512();
    return r;
}

/*
 * acc += v（64-bit向量加到低位，带进位传播到高位）
 * 注意：IFMA 累加器的 hi 部分只有52位有效，不会溢出，
 * 但 add64 用于加 t3 等中间值，需要正确处理进位。
 */
static inline u128_16x u128_16x_add64(u128_16x acc, __m512i v)
{
    /* 使用 AVX-512 的无符号比较来检测进位 */
    __m512i new_lo = _mm512_add_epi64(acc.lo, v);
    /* carry = (new_lo < acc.lo)，使用 _mm512_cmplt_epu64_mask */
    __mmask8 carry_mask = _mm512_cmplt_epu64_mask(new_lo, acc.lo);
    __m512i carry = _mm512_maskz_set1_epi64(carry_mask, 1);
    u128_16x r;
    r.lo = new_lo;
    r.hi = _mm512_add_epi64(acc.hi, carry);
    return r;
}

/*
 * fe_16x_mul：16路并行域乘法
 * r = a * b mod p
 *
 * 使用与 secp256k1_fe_mul_inner 相同的算法，向量化到16路。
 * 利用 AVX-512 IFMA 指令（VPMADD52LUQ/VPMADD52HUQ）实现高效乘加。
 */
static void fe_16x_mul(secp256k1_fe_16x *r,
                       const secp256k1_fe_16x *a,
                       const secp256k1_fe_16x *b)
{
    const uint64_t M_val = 0xFFFFFFFFFFFFFULL;
    const uint64_t R_val = 0x1000003D10ULL;

    /* 为两组（低8路/高8路）分别计算，共循环2次 */
    for (int half = 0; half < 2; half++) {
        __m512i a0 = a->n[0][half], a1 = a->n[1][half],
                a2 = a->n[2][half], a3 = a->n[3][half], a4 = a->n[4][half];
        __m512i b0 = b->n[0][half], b1 = b->n[1][half],
                b2 = b->n[2][half], b3 = b->n[3][half], b4 = b->n[4][half];

        u128_16x c, d;
        __m512i t3, t4, tx, u0;
        __m512i r0, r1, r2, r3, r4;

        /* d = a0*b3 + a1*b2 + a2*b1 + a3*b0 */
        d = u128_16x_zero();
        d = u128_16x_accum_mul(d, a0, b3);
        d = u128_16x_accum_mul(d, a1, b2);
        d = u128_16x_accum_mul(d, a2, b1);
        d = u128_16x_accum_mul(d, a3, b0);

        /* c = a4*b4 */
        c = u128_16x_zero();
        c = u128_16x_accum_mul(c, a4, b4);

        /* d += R * c_lo64; c >>= 64
         * c_lo 最多64位，IFMA只取低52位，需拆分处理：
         *   d += R * c_lo_lo52 + R * c_lo_hi12 * 2^52
         * 后者等价于 d.hi += R * c_lo_hi12（d.hi 对应 d >> 52）
         */
        __m512i mask52_v1 = _mm512_set1_epi64((int64_t)M_val);
        __m512i c_lo = u128_16x_extract64(&c);
        __m512i c_lo_lo = _mm512_and_si512(c_lo, mask52_v1);
        __m512i c_lo_hi = _mm512_srli_epi64(c_lo, 52);
        d = u128_16x_accum_mul_scalar(d, R_val, c_lo_lo);
        __m512i R_vec = _mm512_set1_epi64((int64_t)R_val);
        d.hi = _mm512_madd52lo_epu64(d.hi, R_vec, c_lo_hi);

        /* t3 = d & M; d >>= 52 */
        t3 = u128_16x_extract52(&d);

        /* d += a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 */
        d = u128_16x_accum_mul(d, a0, b4);
        d = u128_16x_accum_mul(d, a1, b3);
        d = u128_16x_accum_mul(d, a2, b2);
        d = u128_16x_accum_mul(d, a3, b1);
        d = u128_16x_accum_mul(d, a4, b0);

        /* d += (R<<12) * c_hi */
        __m512i c_val = c.lo;
        d = u128_16x_accum_mul_scalar(d, R_val << 12, c_val);

        /* t4 = d & M; d >>= 52 */
        t4 = u128_16x_extract52(&d);

        /* tx = t4 >> 48; t4 &= (M >> 4) */
        tx = _mm512_srli_epi64(t4, 48);
        t4 = _mm512_and_si512(t4, _mm512_set1_epi64((int64_t)(M_val >> 4)));

        /* c = a0*b0 */
        c = u128_16x_zero();
        c = u128_16x_accum_mul(c, a0, b0);

        /* d += a1*b4 + a2*b3 + a3*b2 + a4*b1 */
        d = u128_16x_accum_mul(d, a1, b4);
        d = u128_16x_accum_mul(d, a2, b3);
        d = u128_16x_accum_mul(d, a3, b2);
        d = u128_16x_accum_mul(d, a4, b1);

        /* u0 = d & M; d >>= 52 */
        u0 = u128_16x_extract52(&d);

        /* u0 = (u0 << 4) | tx */
        u0 = _mm512_or_si512(_mm512_slli_epi64(u0, 4), tx);

        /* c += u0 * (R >> 4)         * u0 = (u0_52 << 4) | tx，最多 56 位，IFMA 只取低 52 位，需拆分：
         *   c += (u0 & M) * (R>>4) + (u0 >> 52) * (R>>4) * 2^52
         * 后者等价于 c.hi += (u0 >> 52) * (R>>4)
         */
        __m512i mask52_v2 = _mm512_set1_epi64((int64_t)M_val);
        __m512i u0_lo = _mm512_and_si512(u0, mask52_v2);
        __m512i u0_hi = _mm512_srli_epi64(u0, 52);  /* 最多 4 位 */
        c = u128_16x_accum_mul_scalar(c, R_val >> 4, u0_lo);
        __m512i Rdiv4_vec = _mm512_set1_epi64((int64_t)(R_val >> 4));
        c.hi = _mm512_madd52lo_epu64(c.hi, Rdiv4_vec, u0_hi);

        /* r0 = c & M; c >>= 52 */
        r0 = u128_16x_extract52(&c);

        /* c += a0*b1 + a1*b0 */
        c = u128_16x_accum_mul(c, a0, b1);
        c = u128_16x_accum_mul(c, a1, b0);

        /* d += a2*b4 + a3*b3 + a4*b2 */
        d = u128_16x_accum_mul(d, a2, b4);
        d = u128_16x_accum_mul(d, a3, b3);
        d = u128_16x_accum_mul(d, a4, b2);

        /* c += (d & M) * R; d >>= 52 */
        __m512i d_lo = u128_16x_extract52(&d);
        c = u128_16x_accum_mul_scalar(c, R_val, d_lo);

        /* r1 = c & M; c >>= 52 */
        r1 = u128_16x_extract52(&c);

        /* c += a0*b2 + a1*b1 + a2*b0 */
        c = u128_16x_accum_mul(c, a0, b2);
        c = u128_16x_accum_mul(c, a1, b1);
        c = u128_16x_accum_mul(c, a2, b0);

        /* d += a3*b4 + a4*b3 */
        d = u128_16x_accum_mul(d, a3, b4);
        d = u128_16x_accum_mul(d, a4, b3);

        /* c += R * d_lo64; d >>= 64
         * d_lo64 最多64位，IFMA只取低52位，需拆分处理：
         *   c += R * d_lo64_lo52 + R * d_lo64_hi12 * 2^52
         * 后者等价于 c.hi += R * d_lo64_hi12
         */
        __m512i mask52_v3 = _mm512_set1_epi64((int64_t)M_val);
        __m512i d_lo64 = u128_16x_extract64(&d);
        __m512i d_lo64_lo = _mm512_and_si512(d_lo64, mask52_v3);
        __m512i d_lo64_hi = _mm512_srli_epi64(d_lo64, 52);
        c = u128_16x_accum_mul_scalar(c, R_val, d_lo64_lo);
        __m512i R_vec2 = _mm512_set1_epi64((int64_t)R_val);
        c.hi = _mm512_madd52lo_epu64(c.hi, R_vec2, d_lo64_hi);

        /* r2 = c & M; c >>= 52 */
        r2 = u128_16x_extract52(&c);

        /* c += (R<<12) * d_hi + t3 */
        __m512i d_val = d.lo;
        c = u128_16x_accum_mul_scalar(c, R_val << 12, d_val);
        c = u128_16x_add64(c, t3);

        /* r3 = c & M; c >>= 52 */
        r3 = u128_16x_extract52(&c);

        /* r4 = c_lo + t4 */
        r4 = _mm512_add_epi64(c.lo, t4);

        r->n[0][half] = r0;
        r->n[1][half] = r1;
        r->n[2][half] = r2;
        r->n[3][half] = r3;
        r->n[4][half] = r4;
    }
}

/*
 * fe_16x_sqr：16路并行域平方
 * r = a^2 mod p
 */
static void fe_16x_sqr(secp256k1_fe_16x *r, const secp256k1_fe_16x *a)
{
    fe_16x_mul(r, a, a);
}

/*
 * gej_add_ge_var_16way：16路并行 Jacobian + Affine 点加法
 *
 * 等价于对 i=0..15 分别调用：
 *   secp256k1_gej_add_ge_var(&r[i], &a[i], b, &rzr[i])
 *
 * 参数：
 *   r[16]   : 输出 Jacobian 坐标
 *   a[16]   : 输入 Jacobian 坐标（假设非 infinity）
 *   b       : 输入仿射坐标（假设非 infinity）
 *   rzr[16] : 输出 Z 坐标增量因子
 *   normed  : 0=对输入坐标做normalize_weak；1=跳过（调用方保证 magnitude=1）
 */
void gej_add_ge_var_16way(secp256k1_gej r[16],
                          const secp256k1_gej a[16],
                          const secp256k1_ge *b,
                          secp256k1_fe rzr[16],
                          int normed)
{
    /* 加载 a 的 x, y, z 坐标 */
    secp256k1_fe_16x ax, ay, az;
    secp256k1_fe ax_arr[16], ay_arr[16], az_arr[16];
    for (int i = 0; i < 16; i++) {
        ax_arr[i] = a[i].x;
        ay_arr[i] = a[i].y;
        az_arr[i] = a[i].z;
    }
    fe_16x_load(&ax, ax_arr);
    fe_16x_load(&ay, ay_arr);
    fe_16x_load(&az, az_arr);

    /* 若 normed==0，对输入坐标做 normalize_weak，确保 magnitude=1 */
    if (!normed) {
        fe_16x_normalize_weak(&ax);
        fe_16x_normalize_weak(&ay);
        fe_16x_normalize_weak(&az);
    }

    /* 广播 b.x, b.y（仿射坐标，16路共享同一个 b） */
    secp256k1_fe_16x bx, by;
    secp256k1_fe bx_arr[16], by_arr[16];
    for (int i = 0; i < 16; i++) {
        bx_arr[i] = b->x;
        by_arr[i] = b->y;
    }
    fe_16x_load(&bx, bx_arr);
    fe_16x_load(&by, by_arr);

    /* z12 = a.z^2 */
    secp256k1_fe_16x z12;
    fe_16x_sqr(&z12, &az);

    /* u2 = b.x * z12 */
    secp256k1_fe_16x u2;
    fe_16x_mul(&u2, &bx, &z12);

    /* s2 = b.y * z12 * a.z */
    secp256k1_fe_16x s2;
    fe_16x_mul(&s2, &by, &z12);
    fe_16x_mul(&s2, &s2, &az);

    /* h = u2 - u1 */
    secp256k1_fe_16x neg_u1;
    fe_16x_negate(&neg_u1, &ax, 1);
    secp256k1_fe_16x h;
    fe_16x_add(&h, &u2, &neg_u1);
    /* h 的 limb 可能超过 52 位（add+negate 结果），normalize 后才能用于 IFMA 乘法 */
    fe_16x_normalize_weak(&h);

    /* 提前存储 h，用于检测 h=0（a=b 的点倍情况） */
    secp256k1_fe h_arr[16];
    fe_16x_store(h_arr, &h);

    /* i = s1 - s2 */
    secp256k1_fe_16x neg_s2;
    fe_16x_negate(&neg_s2, &s2, 1);
    secp256k1_fe_16x fi;
    fe_16x_add(&fi, &ay, &neg_s2);
    /* fi 的 limb 可能超过 52 位，normalize 后才能用于 IFMA 乘法 */
    fe_16x_normalize_weak(&fi);

    /* r.z = a.z * h */
    secp256k1_fe_16x rz;
    fe_16x_mul(&rz, &az, &h);

    /* rzr = h（已经 normalize_weak，直接使用） */
    secp256k1_fe_16x rzr_16x = h;

    /* h2 = -h^2 */
    secp256k1_fe_16x h2;
    fe_16x_sqr(&h2, &h);
    fe_16x_negate(&h2, &h2, 1);
    /* h2 是 negate 结果，limb 可能超过 52 位，normalize 后才能用于 IFMA 乘法 */
    fe_16x_normalize_weak(&h2);

    /* h3 = h2 * h */
    secp256k1_fe_16x h3;
    fe_16x_mul(&h3, &h2, &h);

    /* t = u1 * h2 */
    secp256k1_fe_16x t;
    fe_16x_mul(&t, &ax, &h2);

    /* r.x = i^2 + h3 + 2*t */
    secp256k1_fe_16x rx;
    fe_16x_sqr(&rx, &fi);
    fe_16x_add(&rx, &rx, &h3);
    fe_16x_add(&rx, &rx, &t);
    fe_16x_add(&rx, &rx, &t);

    /* r.y = (t + r.x) * i + h3 * s1 */
    secp256k1_fe_16x ry;
    secp256k1_fe_16x t2 = t;
    fe_16x_add(&t2, &t2, &rx);
    /* t2 = t + rx，limb 可能超过 52 位，normalize 后才能用于 IFMA 乘法 */
    fe_16x_normalize_weak(&t2);
    fe_16x_mul(&ry, &t2, &fi);
    secp256k1_fe_16x h3s1;
    fe_16x_mul(&h3s1, &h3, &ay);
    fe_16x_add(&ry, &ry, &h3s1);

    /* 归约 rz, rx, ry */
    fe_16x_normalize_weak(&rz);
    fe_16x_normalize_weak(&rx);
    fe_16x_normalize_weak(&ry);

    /* 存储结果 */
    secp256k1_fe rz_arr[16], rx_arr[16], ry_arr[16], rzr_arr[16];
    fe_16x_store(rz_arr, &rz);
    fe_16x_store(rx_arr, &rx);
    fe_16x_store(ry_arr, &ry);
    fe_16x_store(rzr_arr, &rzr_16x);

    for (int i = 0; i < 16; i++) {
        r[i].x = rx_arr[i];
        r[i].y = ry_arr[i];
        r[i].z = rz_arr[i];
        r[i].infinity = 0;
        if (rzr != NULL) {
            rzr[i] = rzr_arr[i];
        }
    }

    /* 修正：对 a[i].infinity == 1 的路，用标量覆盖 */
    for (int i = 0; i < 16; i++) {
        if (a[i].infinity) {
            secp256k1_fe rzr_scalar;
            secp256k1_gej_add_ge_var(&r[i], &a[i], b,
                                     rzr != NULL ? &rzr_scalar : NULL);
            if (rzr != NULL) {
                rzr[i] = rzr_scalar;
            }
        }
    }

    /* 修正：对 h=0（即 a=b，点倍情况）的路，用标量覆盖 */
    for (int i = 0; i < 16; i++) {
        if (secp256k1_fe_normalizes_to_zero(&h_arr[i])) {
            secp256k1_fe rzr_scalar;
            secp256k1_gej_add_ge_var(&r[i], &a[i], b,
                                     rzr != NULL ? &rzr_scalar : NULL);
            if (rzr != NULL) {
                rzr[i] = rzr_scalar;
            }
        }
    }
}

#endif /* __AVX512IFMA__ */
