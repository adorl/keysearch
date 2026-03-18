/*
 * secp256k1_keygen_avx512.c
 *
 * AVX-512 IFMA 16-way parallel finite field arithmetic and point addition implementation.
 *
 * Layout: SoA (Structure of Arrays)
 *   secp256k1_fe_16x.n[i] is a __m512i whose 8 64-bit lanes each store
 *   the i-th limb of 16 points (each lane holds 2 points, 8 lanes x 2 = 16 points).
 *
 * Actual design (used in this implementation):
 *   n[i] is a single __m512i, 8 lanes store 8 of the 16 points (lower 8 lanes)
 *   another __m512i stores the upper 8 lanes, using n[i][0..1] two __m512i.
 *   Total: 5x2=10 __m512i registers.
 *
 * Compilation requirements: -mavx512f -mavx512ifma
 */

#ifdef __AVX512IFMA__

#include "secp256k1_keygen.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* secp256k1_fe_16x and secp256k1_gej_16x are defined in secp256k1_keygen.h */

/*
 * fe_16x_broadcast: broadcast a single secp256k1_fe to all 16 lanes
 * Uses _mm512_set1_epi64 to directly broadcast 5 limbs, avoiding 16-element
 * array fill + load path used by fe_16x_load.
 */
static void fe_16x_broadcast(secp256k1_fe_16x *dst, const secp256k1_fe *a)
{
    for (int i = 0; i < 5; i++) {
        __m512i v = _mm512_set1_epi64((int64_t)a->n[i]);
        dst->n[i][0] = v;
        dst->n[i][1] = v;
    }
}

/*
 * fe_16x_load: convert 16 secp256k1_fe elements to SoA layout
 * Uses aligned buffer + _mm512_load_si512 to avoid _mm512_set_epi64 overhead.
 */
static void fe_16x_load(secp256k1_fe_16x *dst, const secp256k1_fe src[16])
{
    for (int i = 0; i < 5; i++) {
        uint64_t lo[8] __attribute__((aligned(64))) = {
            src[0].n[i], src[1].n[i], src[2].n[i], src[3].n[i],
            src[4].n[i], src[5].n[i], src[6].n[i], src[7].n[i]
        };
        uint64_t hi[8] __attribute__((aligned(64))) = {
            src[8].n[i], src[9].n[i], src[10].n[i], src[11].n[i],
            src[12].n[i], src[13].n[i], src[14].n[i], src[15].n[i]
        };
        dst->n[i][0] = _mm512_load_si512((const void *)lo);
        dst->n[i][1] = _mm512_load_si512((const void *)hi);
    }
}

/*
 * fe_16x_load_ptrs: load 16 secp256k1_fe from scattered pointers into SoA layout.
 * Same as fe_16x_load but accepts an array of 16 pointers instead of a contiguous array,
 * allowing direct loading from non-contiguous memory (e.g. ge_buf[lane][step].x)
 * without an intermediate copy.
 */
void fe_16x_load_ptrs(secp256k1_fe_16x *dst, const secp256k1_fe *src_ptrs[16])
{
    for (int i = 0; i < 5; i++) {
        uint64_t lo[8] __attribute__((aligned(64))) = {
            src_ptrs[0]->n[i], src_ptrs[1]->n[i], src_ptrs[2]->n[i], src_ptrs[3]->n[i],
            src_ptrs[4]->n[i], src_ptrs[5]->n[i], src_ptrs[6]->n[i], src_ptrs[7]->n[i]
        };
        uint64_t hi[8] __attribute__((aligned(64))) = {
            src_ptrs[8]->n[i], src_ptrs[9]->n[i], src_ptrs[10]->n[i], src_ptrs[11]->n[i],
            src_ptrs[12]->n[i], src_ptrs[13]->n[i], src_ptrs[14]->n[i], src_ptrs[15]->n[i]
        };
        dst->n[i][0] = _mm512_load_si512((const void *)lo);
        dst->n[i][1] = _mm512_load_si512((const void *)hi);
    }
}

/*
 * fe_16x_store: convert SoA layout back to 16 secp256k1_fe elements (contiguous array)
 *
 * Optimized write pattern: process one limb at a time, store to a single
 * aligned buffer, then immediately scatter to destination elements.
 * This avoids the [5][8] double-buffer random-read pattern and keeps
 * writes sequential per destination element, reducing cache-line conflicts.
 */
static void fe_16x_store(secp256k1_fe dst[16], const secp256k1_fe_16x *src)
{
    uint64_t buf[8] __attribute__((aligned(64)));
    for (int i = 0; i < 5; i++) {
        _mm512_store_si512((__m512i *)buf, src->n[i][0]);
        for (int j = 0; j < 8; j++)
            dst[j].n[i] = buf[j];
        _mm512_store_si512((__m512i *)buf, src->n[i][1]);
        for (int j = 0; j < 8; j++)
            dst[j + 8].n[i] = buf[j];
    }
}

/*
 * gej_16x_load: convert 16 secp256k1_gej elements (AoS) to SoA layout
 */
void gej_16x_load(secp256k1_gej_16x *dst, const secp256k1_gej src[16])
{
    secp256k1_fe x_arr[16], y_arr[16], z_arr[16];
    for (int i = 0; i < 16; i++) {
        x_arr[i] = src[i].x;
        y_arr[i] = src[i].y;
        z_arr[i] = src[i].z;
    }
    fe_16x_load(&dst->x, x_arr);
    fe_16x_load(&dst->y, y_arr);
    fe_16x_load(&dst->z, z_arr);
}

/*
 * gej_16x_store: convert SoA layout back to 16 secp256k1_gej elements (AoS)
 */
void gej_16x_store(secp256k1_gej dst[16], const secp256k1_gej_16x *src)
{
    secp256k1_fe x_arr[16], y_arr[16], z_arr[16];
    fe_16x_store(x_arr, &src->x);
    fe_16x_store(y_arr, &src->y);
    fe_16x_store(z_arr, &src->z);
    for (int i = 0; i < 16; i++) {
        dst[i].x = x_arr[i];
        dst[i].y = y_arr[i];
        dst[i].z = z_arr[i];
        dst[i].infinity = 0;
    }
}

/*
 * gej_16x_store_scatter: convert SoA layout and scatter-write to 16 non-contiguous
 * secp256k1_gej pointers. Used to write directly into gej_buf[ch][step] without
 * an intermediate contiguous array, eliminating one full 2048-byte copy per step.
 */
void gej_16x_store_scatter(secp256k1_gej *dst_ptrs[16], const secp256k1_gej_16x *src)
{
    secp256k1_fe x_arr[16], y_arr[16], z_arr[16];
    fe_16x_store(x_arr, &src->x);
    fe_16x_store(y_arr, &src->y);
    fe_16x_store(z_arr, &src->z);
    for (int i = 0; i < 16; i++) {
        dst_ptrs[i]->x = x_arr[i];
        dst_ptrs[i]->y = y_arr[i];
        dst_ptrs[i]->z = z_arr[i];
        dst_ptrs[i]->infinity = 0;
    }
}

/*
 * Basic field operations
 */

/*
 * fe_16x_add: 16-way parallel field addition (no modular reduction, magnitude accumulates)
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
 * fe_16x_negate: 16-way parallel field negation
 * r = -a, input magnitude m, output magnitude m+1
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
 * fe_16x_normalize_weak: 16-way parallel weak normalization
 * Reduces magnitude to 1
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
         * x is at most 4 bits, 0x1000003D1 = (1<<32) + 0x3D1
         * = (x << 32) + x * 0x3D1
         * Use _mm512_mul_epu32 to compute low 32-bit product (x <= 0xF, 0x3D1 <= 0xFFF, product <= 16 bits, safe) */
        __m512i R1_lo = _mm512_set1_epi64(0x3D1LL);
        __m512i xR = _mm512_add_epi64(
            _mm512_slli_epi64(x, 32),
            _mm512_mul_epu32(x, R1_lo)
        );
        t0 = _mm512_add_epi64(t0, xR);

        /* Carry propagation */
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
 * AVX-512 IFMA field multiplication
 */

/*
 * 128-bit vector accumulator: (hi, lo) represents hi*2^64 + lo
 * Uses a pair of __m512i, each lane independently maintains a 128-bit accumulator
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
 * acc += a * b (using AVX-512 IFMA instructions: VPMADD52LUQ / VPMADD52HUQ)
 * a, b are both 52-bit limbs (stored in 64-bit lanes)
 * VPMADD52LUQ: acc.lo += (a * b) & ((1<<52)-1)  (low 52 bits)
 * VPMADD52HUQ: acc.hi += (a * b) >> 52           (high 52 bits)
 */
static inline u128_16x u128_16x_accum_mul(u128_16x acc, __m512i a, __m512i b)
{
    u128_16x r;
    r.lo = _mm512_madd52lo_epu64(acc.lo, a, b);
    r.hi = _mm512_madd52hi_epu64(acc.hi, a, b);
    return r;
}

/*
 * acc += scalar * b (scalar is a broadcast 64-bit constant)
 */
static inline u128_16x u128_16x_accum_mul_scalar(u128_16x acc, uint64_t scalar, __m512i b)
{
    __m512i s = _mm512_set1_epi64((int64_t)scalar);
    return u128_16x_accum_mul(acc, s, b);
}

/*
 * Extract low 52 bits of acc, return the low 52-bit value, acc >>= 52
 *
 * True value of acc = acc.lo + acc.hi * 2^52
 * Low 52 bits = acc.lo & M
 * After right shift by 52, new true value = (acc.lo >> 52) + acc.hi
 * Therefore:
 *   new acc.lo = acc.hi + (acc.lo >> 52)
 *   new acc.hi = 0
 *
 * Note: acc.hi stores the acc >> 52 portion; after shifting right by 52,
 * acc.hi directly becomes the new low part, plus the overflow of acc.lo (acc.lo >> 52, at most 12 bits).
 */
static inline __m512i u128_16x_extract52(u128_16x *acc)
{
    __m512i mask52 = _mm512_set1_epi64((int64_t)0xFFFFFFFFFFFFFULL);
    __m512i r = _mm512_and_si512(acc->lo, mask52);
    /* acc >>= 52: new true value = acc.hi + (acc.lo >> 52) */
    __m512i lo_carry = _mm512_srli_epi64(acc->lo, 52);  /* acc.lo >> 52, at most 12 bits */
    acc->lo = _mm512_add_epi64(acc->hi, lo_carry);
    acc->hi = _mm512_setzero_si512();
    return r;
}

/*
 * Extract low 64 bits of acc, acc >>= 64
 *
 * True value = acc.lo + acc.hi * 2^52
 * Low 64 bits = acc.lo + (acc.hi & 0xFFF) * 2^52  (mod 2^64, 64-bit truncation)
 * After right shift by 64 = acc.hi >> 12 + carry
 * where carry = 1 if addition overflows 64 bits, else 0
 */
static inline __m512i u128_16x_extract64(u128_16x *acc)
{
    __m512i mask12 = _mm512_set1_epi64(0xFFFLL);
    __m512i acc_hi_lo12 = _mm512_and_si512(acc->hi, mask12);
    __m512i acc_hi_hi   = _mm512_srli_epi64(acc->hi, 12);

    /* r = acc.lo + (acc.hi & 0xFFF) * 2^52 (low 64 bits, auto-truncated) */
    __m512i hi_contrib = _mm512_slli_epi64(acc_hi_lo12, 52);
    __m512i r = _mm512_add_epi64(acc->lo, hi_contrib);

    /* carry = (r < acc.lo), i.e. whether addition overflowed */
    __mmask8 carry_mask = _mm512_cmplt_epu64_mask(r, acc->lo);
    __m512i carry = _mm512_maskz_set1_epi64(carry_mask, 1);

    acc->lo = _mm512_add_epi64(acc_hi_hi, carry);
    acc->hi = _mm512_setzero_si512();
    return r;
}

/*
 * acc += v (add 64-bit vector to low part, with carry propagation to high part)
 * Note: the hi part of the IFMA accumulator has only 52 valid bits and won't overflow,
 * but add64 is used to add intermediate values like t3, requiring correct carry handling.
 */
static inline u128_16x u128_16x_add64(u128_16x acc, __m512i v)
{
    /* Use AVX-512 unsigned comparison to detect carry */
    __m512i new_lo = _mm512_add_epi64(acc.lo, v);
    /* carry = (new_lo < acc.lo), using _mm512_cmplt_epu64_mask */
    __mmask8 carry_mask = _mm512_cmplt_epu64_mask(new_lo, acc.lo);
    __m512i carry = _mm512_maskz_set1_epi64(carry_mask, 1);
    u128_16x r;
    r.lo = new_lo;
    r.hi = _mm512_add_epi64(acc.hi, carry);
    return r;
}

/*
 * fe_16x_mul: 16-way parallel field multiplication
 * r = a * b mod p
 *
 * Uses the same algorithm as secp256k1_fe_mul_inner, vectorized to 16 lanes.
 * Leverages AVX-512 IFMA instructions (VPMADD52LUQ/VPMADD52HUQ) for efficient multiply-add.
 */
static void fe_16x_mul(secp256k1_fe_16x *r,
                       const secp256k1_fe_16x *a,
                       const secp256k1_fe_16x *b)
{
    const uint64_t M_val = 0xFFFFFFFFFFFFFULL;
    const uint64_t R_val = 0x1000003D10ULL;

    /* Hoist constant vectors out of the half loop to avoid redundant construction */
    const __m512i mask52_v = _mm512_set1_epi64((int64_t)M_val);
    const __m512i R_vec = _mm512_set1_epi64((int64_t)R_val);
    const __m512i Rdiv4_vec = _mm512_set1_epi64((int64_t)(R_val >> 4));
    const __m512i mask48_v = _mm512_set1_epi64((int64_t)(M_val >> 4));

    /* Compute separately for two halves (lower 8 / upper 8 lanes), loop twice */
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
         * c_lo is at most 64 bits, IFMA only takes low 52 bits, must split:
         *   d += R * c_lo_lo52 + R * c_lo_hi12 * 2^52
         * The latter is equivalent to d.hi += R * c_lo_hi12 (d.hi corresponds to d >> 52)
         */
        __m512i c_lo = u128_16x_extract64(&c);
        __m512i c_lo_lo = _mm512_and_si512(c_lo, mask52_v);
        __m512i c_lo_hi = _mm512_srli_epi64(c_lo, 52);
        d = u128_16x_accum_mul_scalar(d, R_val, c_lo_lo);
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
        t4 = _mm512_and_si512(t4, mask48_v);

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

        /* c += u0 * (R >> 4)
         * u0 = (u0_52 << 4) | tx, at most 56 bits, IFMA only takes low 52 bits, must split:
         *   c += (u0 & M) * (R>>4) + (u0 >> 52) * (R>>4) * 2^52
         * The latter is equivalent to c.hi += (u0 >> 52) * (R>>4)
         */
        __m512i u0_lo = _mm512_and_si512(u0, mask52_v);
        __m512i u0_hi = _mm512_srli_epi64(u0, 52);  /* at most 4 bits */
        c = u128_16x_accum_mul_scalar(c, R_val >> 4, u0_lo);
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
         * d_lo64 is at most 64 bits, IFMA only takes low 52 bits, must split:
         *   c += R * d_lo64_lo52 + R * d_lo64_hi12 * 2^52
         * The latter is equivalent to c.hi += R * d_lo64_hi12
         */
        __m512i d_lo64 = u128_16x_extract64(&d);
        __m512i d_lo64_lo = _mm512_and_si512(d_lo64, mask52_v);
        __m512i d_lo64_hi = _mm512_srli_epi64(d_lo64, 52);
        c = u128_16x_accum_mul_scalar(c, R_val, d_lo64_lo);
        c.hi = _mm512_madd52lo_epu64(c.hi, R_vec, d_lo64_hi);

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
 * fe_16x_sqr: 16-way parallel field squaring
 * r = a^2 mod p
 *
 * Specialized squaring exploiting a[i]*a[j] == a[j]*a[i] symmetry.
 * Cross-terms are computed once and doubled via pre-multiplied 2*a[i],
 * reducing IFMA multiply-add ops from 25 (in generic mul) to ~15.
 *
 * Same algorithm structure as fe_16x_mul, but with symmetric optimization.
 */
static void fe_16x_sqr(secp256k1_fe_16x *r, const secp256k1_fe_16x *a)
{
    const uint64_t M_val = 0xFFFFFFFFFFFFFULL;
    const uint64_t R_val = 0x1000003D10ULL;

    for (int half = 0; half < 2; half++) {
        __m512i a0 = a->n[0][half], a1 = a->n[1][half],
                a2 = a->n[2][half], a3 = a->n[3][half], a4 = a->n[4][half];

        /*
         * IFMA (VPMADD52) only uses the low 52 bits of each operand.
         * Since limbs can be up to 52 bits, pre-multiplying by 2 (shift left 1)
         * would produce 53-bit values whose MSB gets truncated by IFMA.
         *
         * Fix: double cross-terms by accumulating twice instead of pre-shifting.
         * e.g. 2*(a0*a3) = accum(a0,a3) + accum(a0,a3) instead of accum(a0<<1, a3).
         */

        u128_16x c, d;
        __m512i t3, t4, tx, u0;
        __m512i r0, r1, r2, r3, r4;
        __m512i mask52_v = _mm512_set1_epi64((int64_t)M_val);
        __m512i R_vec = _mm512_set1_epi64((int64_t)R_val);

        /* d = 2*(a0*a3) + 2*(a1*a2) */
        d = u128_16x_zero();
        d = u128_16x_accum_mul(d, a0, a3);
        d = u128_16x_accum_mul(d, a0, a3);
        d = u128_16x_accum_mul(d, a1, a2);
        d = u128_16x_accum_mul(d, a1, a2);

        /* c = a4*a4 */
        c = u128_16x_zero();
        c = u128_16x_accum_mul(c, a4, a4);

        /* d += R * c_lo64; c >>= 64 */
        __m512i c_lo = u128_16x_extract64(&c);
        __m512i c_lo_lo = _mm512_and_si512(c_lo, mask52_v);
        __m512i c_lo_hi = _mm512_srli_epi64(c_lo, 52);
        d = u128_16x_accum_mul_scalar(d, R_val, c_lo_lo);
        d.hi = _mm512_madd52lo_epu64(d.hi, R_vec, c_lo_hi);

        /* t3 = d & M; d >>= 52 */
        t3 = u128_16x_extract52(&d);

        /* d += 2*(a0*a4) + 2*(a1*a3) + a2*a2 */
        d = u128_16x_accum_mul(d, a0, a4);
        d = u128_16x_accum_mul(d, a0, a4);
        d = u128_16x_accum_mul(d, a1, a3);
        d = u128_16x_accum_mul(d, a1, a3);
        d = u128_16x_accum_mul(d, a2, a2);

        /* d += (R<<12) * c_hi */
        __m512i c_val = c.lo;
        d = u128_16x_accum_mul_scalar(d, R_val << 12, c_val);

        /* t4 = d & M; d >>= 52 */
        t4 = u128_16x_extract52(&d);

        /* tx = t4 >> 48; t4 &= (M >> 4) */
        tx = _mm512_srli_epi64(t4, 48);
        t4 = _mm512_and_si512(t4, _mm512_set1_epi64((int64_t)(M_val >> 4)));

        /* c = a0*a0 */
        c = u128_16x_zero();
        c = u128_16x_accum_mul(c, a0, a0);

        /* d += 2*(a1*a4) + 2*(a2*a3) */
        d = u128_16x_accum_mul(d, a1, a4);
        d = u128_16x_accum_mul(d, a1, a4);
        d = u128_16x_accum_mul(d, a2, a3);
        d = u128_16x_accum_mul(d, a2, a3);

        /* u0 = d & M; d >>= 52 */
        u0 = u128_16x_extract52(&d);

        /* u0 = (u0 << 4) | tx */
        u0 = _mm512_or_si512(_mm512_slli_epi64(u0, 4), tx);

        /* c += u0 * (R >> 4), split for >52-bit u0 */
        __m512i u0_lo = _mm512_and_si512(u0, mask52_v);
        __m512i u0_hi = _mm512_srli_epi64(u0, 52);
        c = u128_16x_accum_mul_scalar(c, R_val >> 4, u0_lo);
        __m512i Rdiv4_vec = _mm512_set1_epi64((int64_t)(R_val >> 4));
        c.hi = _mm512_madd52lo_epu64(c.hi, Rdiv4_vec, u0_hi);

        /* r0 = c & M; c >>= 52 */
        r0 = u128_16x_extract52(&c);

        /* c += 2*(a0*a1) */
        c = u128_16x_accum_mul(c, a0, a1);
        c = u128_16x_accum_mul(c, a0, a1);

        /* d += 2*(a2*a4) + a3*a3 */
        d = u128_16x_accum_mul(d, a2, a4);
        d = u128_16x_accum_mul(d, a2, a4);
        d = u128_16x_accum_mul(d, a3, a3);

        /* c += (d & M) * R; d >>= 52 */
        __m512i d_lo = u128_16x_extract52(&d);
        c = u128_16x_accum_mul_scalar(c, R_val, d_lo);

        /* r1 = c & M; c >>= 52 */
        r1 = u128_16x_extract52(&c);

        /* c += 2*(a0*a2) + a1*a1 */
        c = u128_16x_accum_mul(c, a0, a2);
        c = u128_16x_accum_mul(c, a0, a2);
        c = u128_16x_accum_mul(c, a1, a1);

        /* d += 2*(a3*a4) */
        d = u128_16x_accum_mul(d, a3, a4);
        d = u128_16x_accum_mul(d, a3, a4);

        /* c += R * d_lo64; d >>= 64 */
        __m512i d_lo64 = u128_16x_extract64(&d);
        __m512i d_lo64_lo = _mm512_and_si512(d_lo64, mask52_v);
        __m512i d_lo64_hi = _mm512_srli_epi64(d_lo64, 52);
        c = u128_16x_accum_mul_scalar(c, R_val, d_lo64_lo);
        c.hi = _mm512_madd52lo_epu64(c.hi, R_vec, d_lo64_hi);

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
 * gej_add_ge_var_16way: 16-way parallel Jacobian + Affine point addition
 *
 * Equivalent to calling for each i=0..15:
 *   secp256k1_gej_add_ge_var(&r[i], &a[i], b, &rzr[i])
 *
 * Parameters:
 *   r[16]   : output Jacobian coordinates
 *   a[16]   : input Jacobian coordinates (assumed non-infinity)
 *   b       : input affine coordinates (assumed non-infinity)
 *   rzr[16] : output Z coordinate ratio factors
 *   normed  : 0=apply normalize_weak to input coordinates; 1=skip (caller guarantees magnitude=1)
 */
void gej_add_ge_var_16way(secp256k1_gej r[16],
                          const secp256k1_gej a[16],
                          const secp256k1_ge *b,
                          secp256k1_fe rzr[16],
                          int normed)
{
    /* Load x, y, z coordinates of a */
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

    /* If normed==0, apply normalize_weak to input coordinates to ensure magnitude=1 */
    if (!normed) {
        fe_16x_normalize_weak(&ax);
        fe_16x_normalize_weak(&ay);
        fe_16x_normalize_weak(&az);
    }

    /* Broadcast b.x, b.y (affine coordinates, shared by all 16 lanes) */
    secp256k1_fe_16x bx, by;
    fe_16x_broadcast(&bx, &b->x);
    fe_16x_broadcast(&by, &b->y);

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
    /* h limbs may exceed 52 bits (result of add+negate), must normalize before IFMA multiply */
    fe_16x_normalize_weak(&h);

    /* Store h early for detecting h=0 (point doubling case when a=b) */
    secp256k1_fe h_arr[16];
    fe_16x_store(h_arr, &h);

    /* i = s1 - s2 */
    secp256k1_fe_16x neg_s2;
    fe_16x_negate(&neg_s2, &s2, 1);
    secp256k1_fe_16x fi;
    fe_16x_add(&fi, &ay, &neg_s2);
    /* fi limbs may exceed 52 bits, must normalize before IFMA multiply */
    fe_16x_normalize_weak(&fi);

    /* r.z = a.z * h */
    secp256k1_fe_16x rz;
    fe_16x_mul(&rz, &az, &h);

    /* rzr = h (already normalize_weak, use directly) */
    secp256k1_fe_16x rzr_16x = h;

    /* h2 = -h^2 */
    secp256k1_fe_16x h2;
    fe_16x_sqr(&h2, &h);
    fe_16x_negate(&h2, &h2, 1);
    /* h2 is a negate result, limbs may exceed 52 bits, must normalize before IFMA multiply */
    fe_16x_normalize_weak(&h2);

    /* h3 = h2 * h */
    secp256k1_fe_16x h3;
    fe_16x_mul(&h3, &h2, &h);

    /* t = u1 * h2 */
    secp256k1_fe_16x t;
    fe_16x_mul(&t, &ax, &h2);

    /* r.x = i^2 + h3 + 2*t
     * fi^2 output magnitude=1, h3 magnitude=1, t magnitude=1
     * After 3 additions: magnitude=4, still within safe range (< 2^56) */
    secp256k1_fe_16x rx;
    fe_16x_sqr(&rx, &fi);
    fe_16x_add(&rx, &rx, &h3);
    fe_16x_add(&rx, &rx, &t);
    fe_16x_add(&rx, &rx, &t);

    /* r.y = (t + r.x) * i + h3 * s1
     * t2 = t + rx: t magnitude=1, rx magnitude=4 => t2 magnitude=5
     * Must normalize before IFMA multiply (magnitude must be <= 1) */
    secp256k1_fe_16x ry;
    secp256k1_fe_16x t2 = t;
    fe_16x_add(&t2, &t2, &rx);
    fe_16x_normalize_weak(&t2);
    fe_16x_mul(&ry, &t2, &fi);
    secp256k1_fe_16x h3s1;
    fe_16x_mul(&h3s1, &h3, &ay);
    fe_16x_add(&ry, &ry, &h3s1);

    /*
     * Output normalization: rz, rx, ry are fed back as inputs to next
     * gej_add_ge_var_16way (when normed=1). fe_16x_mul output has
     * magnitude=1 which is fine for IFMA. Only rx (magnitude=4) and
     * ry (magnitude=2, from mul+add) need normalize_weak.
     * rz was produced by fe_16x_mul, magnitude=1, skip normalize.
     */
    fe_16x_normalize_weak(&rx);
    fe_16x_normalize_weak(&ry);

    /* Store results */
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

    /* Fixup: for lanes where a[i].infinity == 1, overwrite with scalar result */
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

    /* Fixup: for lanes where h=0 (i.e. a=b, point doubling case), overwrite with scalar result */
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

/*
 * gej_add_ge_var_16way_soa: 16-way parallel Jacobian + Affine point addition (SoA persistent)
 *
 * Same algorithm as gej_add_ge_var_16way, but operates directly on SoA layout.
 * Input/output are secp256k1_gej_16x (SoA), eliminating AoS<->SoA conversion
 * in the hot loop between consecutive calls.
 *
 * Parameters:
 *   r_soa     : output Jacobian coordinates (SoA)
 *   a_soa     : input Jacobian coordinates (SoA, assumed non-infinity)
 *   b         : input affine coordinates (assumed non-infinity)
 *   r_aos_ptrs: array of 16 pointers to output AoS Jacobian coordinates (scatter write);
 *               may be NULL to skip AoS output entirely, individual pointers must not be NULL
 *   rzr_out   : output Z coordinate ratio factors (AoS, 16 elements)
 *   normed    : 0=apply normalize_weak to input coordinates; 1=skip
 */
void gej_add_ge_var_16way_soa(secp256k1_gej_16x *r_soa,
                              const secp256k1_gej_16x *a_soa,
                              const secp256k1_ge *b,
                              secp256k1_gej *r_aos_ptrs[16],
                              secp256k1_fe rzr_out[16],
                              int normed)
{
    secp256k1_fe_16x ax = a_soa->x, ay = a_soa->y, az = a_soa->z;

    /* If normed==0, apply normalize_weak to input coordinates to ensure magnitude=1 */
    if (!normed) {
        fe_16x_normalize_weak(&ax);
        fe_16x_normalize_weak(&ay);
        fe_16x_normalize_weak(&az);
    }

    /* Broadcast b.x, b.y (affine coordinates, shared by all 16 lanes) */
    secp256k1_fe_16x bx, by;
    fe_16x_broadcast(&bx, &b->x);
    fe_16x_broadcast(&by, &b->y);

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
    fe_16x_normalize_weak(&h);

    /*
     * Detect h==0 (mod P) lanes using SIMD.
     *
     * After normalize_weak, each limb is in [0, 2^52) and carries are resolved.
     * If h==0 (mod P), the representation is either exactly 0 or exactly P:
     *   zero: all limbs == 0
     *   P:    n[0]=0xFFFFEFFFFFC2F, n[1..3]=0xFFFFFFFFFFFFF, n[4]=0x0FFFFFFFFFFFF
     *
     * Check both conditions by OR-ing all limbs (==0 test) and by comparing
     * each limb against P's constants (==P test). This avoids the carry
     * propagation that secp256k1_fe_impl_normalizes_to_zero performs.
     */
    __mmask8 h_zero_mask_lo, h_zero_mask_hi;
    __m512i P0 = _mm512_set1_epi64((int64_t)0xFFFFEFFFFFC2FULL);
    __m512i P1 = _mm512_set1_epi64((int64_t)0xFFFFFFFFFFFFFULL);
    __m512i P4 = _mm512_set1_epi64((int64_t)0x0FFFFFFFFFFFFULL);

    for (int half = 0; half < 2; half++) {
        __m512i n0 = h.n[0][half], n1 = h.n[1][half],
                n2 = h.n[2][half], n3 = h.n[3][half], n4 = h.n[4][half];

        /* Test ==0: OR all limbs, then compare to zero */
        __m512i z = _mm512_or_si512(_mm512_or_si512(n0, n1),
                        _mm512_or_si512(_mm512_or_si512(n2, n3), n4));
        __mmask8 is_zero = _mm512_cmpeq_epi64_mask(z, _mm512_setzero_si512());

        /* Test ==P: AND per-limb equality masks */
        __mmask8 is_P = _mm512_cmpeq_epi64_mask(n0, P0)
                        & _mm512_cmpeq_epi64_mask(n1, P1)
                        & _mm512_cmpeq_epi64_mask(n2, P1)
                        & _mm512_cmpeq_epi64_mask(n3, P1)
                        & _mm512_cmpeq_epi64_mask(n4, P4);

        if (half == 0)
            h_zero_mask_lo = is_zero | is_P;
        else
            h_zero_mask_hi = is_zero | is_P;
    }
    /* Combined 16-bit mask: lo in bits 0..7, hi in bits 8..15 */
    uint16_t h_zero_mask = (uint16_t)h_zero_mask_lo | ((uint16_t)h_zero_mask_hi << 8);

    /* i = s1 - s2 */
    secp256k1_fe_16x neg_s2;
    fe_16x_negate(&neg_s2, &s2, 1);
    secp256k1_fe_16x fi;
    fe_16x_add(&fi, &ay, &neg_s2);
    fe_16x_normalize_weak(&fi);

    /* r.z = a.z * h */
    secp256k1_fe_16x rz;
    fe_16x_mul(&rz, &az, &h);

    /* rzr = h */
    secp256k1_fe_16x rzr_16x = h;

    /* h2 = -h^2 */
    secp256k1_fe_16x h2;
    fe_16x_sqr(&h2, &h);
    fe_16x_negate(&h2, &h2, 1);
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
    fe_16x_normalize_weak(&t2);
    fe_16x_mul(&ry, &t2, &fi);
    secp256k1_fe_16x h3s1;
    fe_16x_mul(&h3s1, &h3, &ay);
    fe_16x_add(&ry, &ry, &h3s1);

    /* Output normalization */
    fe_16x_normalize_weak(&rx);
    fe_16x_normalize_weak(&ry);

    /* Store SoA result for persistent use */
    r_soa->x = rx;
    r_soa->y = ry;
    r_soa->z = rz;

    /* Store AoS results for gej_buf (scatter write) if requested */
    if (r_aos_ptrs != NULL) {
        gej_16x_store_scatter(r_aos_ptrs, r_soa);
    }
    if (rzr_out != NULL) {
        fe_16x_store(rzr_out, &rzr_16x);
    }

    /* Fixup: for lanes where h=0 (point doubling case), overwrite with scalar result.
     * h_zero_mask was computed above using SIMD normalizes_to_zero.
     * This is extremely rare (h=0 means a==b), so the scalar fallback is acceptable.
     * Note: infinity fixup is not needed here since the caller handles initial loading
     * through gej_16x_load which only processes non-infinity points */
    if (h_zero_mask != 0 && r_aos_ptrs != NULL) {
        /* Reconstruct AoS only once for all fixup lanes (lazy, only if needed) */
        secp256k1_fe ax_arr[16], ay_arr[16], az_arr[16];
        fe_16x_store(ax_arr, &a_soa->x);
        fe_16x_store(ay_arr, &a_soa->y);
        fe_16x_store(az_arr, &a_soa->z);
        uint16_t mask = h_zero_mask;
        while (mask) {
            int i = __builtin_ctz(mask);
            mask &= mask - 1;
            secp256k1_fe rzr_scalar;
            secp256k1_gej a_lane;
            a_lane.x = ax_arr[i];
            a_lane.y = ay_arr[i];
            a_lane.z = az_arr[i];
            a_lane.infinity = 0;
            secp256k1_gej_add_ge_var(r_aos_ptrs[i], &a_lane, b,
                                     rzr_out != NULL ? &rzr_scalar : NULL);
            if (rzr_out != NULL) {
                rzr_out[i] = rzr_scalar;
            }
        }
    }
}

/*
 * fe_get_b32_16way: batch convert 16 normalized secp256k1_fe to 32-byte big-endian.
 *
 * Uses AVX-512 to process 8 field elements at a time (2 halves for 16 total).
 * For each limb, loads 8 elements into a __m512i, extracts bytes via shift+mask,
 * and scatters to output buffers. This replaces 16 scalar secp256k1_fe_impl_get_b32
 * calls with batched SIMD extraction.
 *
 * The 5x52-bit limb layout maps to 32 bytes as:
 *   n[4]: bits 208..255 -> r[0..5]   (48 bits = 6 bytes)
 *   n[3]: bits 156..207 -> r[6..12]  (52 bits, but byte 12 is split with n[2])
 *   n[2]: bits 104..155 -> r[12..18] (byte 12 upper nibble from n[3])
 *   n[1]: bits 52..103  -> r[19..25] (byte 25 split with n[0])
 *   n[0]: bits 0..51    -> r[25..31] (byte 25 upper nibble from n[1])
 */
void fe_get_b32_16way(uint8_t out[16][32], const secp256k1_fe fe_arr[16])
{
    /*
     * Process in two halves of 8 elements each.
     * For each half, load limbs into __m512i (8 x uint64), then extract bytes
     * using aligned store + scalar scatter. This is faster than 8 independent
     * scalar calls because we amortize the loop overhead and keep data in cache.
     */
    uint64_t buf[8] __attribute__((aligned(64)));

    for (int half = 0; half < 2; half++) {
        const secp256k1_fe *src = fe_arr + half * 8;
        uint8_t (*dst)[32] = out + half * 8;

        /* Load and process each limb with SIMD gather, then scatter bytes */
        /* n[4]: 48 bits -> bytes 0..5 */
        {
            __m512i v4 = _mm512_set_epi64(
                (long long)src[7].n[4], (long long)src[6].n[4],
                (long long)src[5].n[4], (long long)src[4].n[4],
                (long long)src[3].n[4], (long long)src[2].n[4],
                (long long)src[1].n[4], (long long)src[0].n[4]);
            _mm512_store_si512((__m512i *)buf, v4);
            for (int j = 0; j < 8; j++) {
                uint64_t w = buf[j];
                dst[j][0] = (uint8_t)(w >> 40);
                dst[j][1] = (uint8_t)(w >> 32);
                dst[j][2] = (uint8_t)(w >> 24);
                dst[j][3] = (uint8_t)(w >> 16);
                dst[j][4] = (uint8_t)(w >> 8);
                dst[j][5] = (uint8_t)(w);
            }
        }
        /* n[3]: 52 bits -> bytes 6..11, byte 12 lower nibble */
        {
            __m512i v3 = _mm512_set_epi64(
                (long long)src[7].n[3], (long long)src[6].n[3],
                (long long)src[5].n[3], (long long)src[4].n[3],
                (long long)src[3].n[3], (long long)src[2].n[3],
                (long long)src[1].n[3], (long long)src[0].n[3]);
            _mm512_store_si512((__m512i *)buf, v3);
            for (int j = 0; j < 8; j++) {
                uint64_t w = buf[j];
                dst[j][6]  = (uint8_t)(w >> 44);
                dst[j][7]  = (uint8_t)(w >> 36);
                dst[j][8]  = (uint8_t)(w >> 28);
                dst[j][9]  = (uint8_t)(w >> 20);
                dst[j][10] = (uint8_t)(w >> 12);
                dst[j][11] = (uint8_t)(w >> 4);
                /* byte 12: upper nibble from n[2], lower nibble from n[3] */
                dst[j][12] = (uint8_t)((w & 0xF) << 4); /* lower nibble, will OR with n[2] part */
            }
        }
        /* n[2]: 52 bits -> byte 12 upper nibble, bytes 13..18 */
        {
            __m512i v2 = _mm512_set_epi64(
                (long long)src[7].n[2], (long long)src[6].n[2],
                (long long)src[5].n[2], (long long)src[4].n[2],
                (long long)src[3].n[2], (long long)src[2].n[2],
                (long long)src[1].n[2], (long long)src[0].n[2]);
            _mm512_store_si512((__m512i *)buf, v2);
            for (int j = 0; j < 8; j++) {
                uint64_t w = buf[j];
                dst[j][12] |= (uint8_t)((w >> 48) & 0x0F); /* OR upper nibble */
                dst[j][13] = (uint8_t)(w >> 40);
                dst[j][14] = (uint8_t)(w >> 32);
                dst[j][15] = (uint8_t)(w >> 24);
                dst[j][16] = (uint8_t)(w >> 16);
                dst[j][17] = (uint8_t)(w >> 8);
                dst[j][18] = (uint8_t)(w);
            }
        }
        /* n[1]: 52 bits -> bytes 19..24, byte 25 lower nibble */
        {
            __m512i v1 = _mm512_set_epi64(
                (long long)src[7].n[1], (long long)src[6].n[1],
                (long long)src[5].n[1], (long long)src[4].n[1],
                (long long)src[3].n[1], (long long)src[2].n[1],
                (long long)src[1].n[1], (long long)src[0].n[1]);
            _mm512_store_si512((__m512i *)buf, v1);
            for (int j = 0; j < 8; j++) {
                uint64_t w = buf[j];
                dst[j][19] = (uint8_t)(w >> 44);
                dst[j][20] = (uint8_t)(w >> 36);
                dst[j][21] = (uint8_t)(w >> 28);
                dst[j][22] = (uint8_t)(w >> 20);
                dst[j][23] = (uint8_t)(w >> 12);
                dst[j][24] = (uint8_t)(w >> 4);
                dst[j][25] = (uint8_t)((w & 0xF) << 4); /* lower nibble, will OR with n[0] part */
            }
        }
        /* n[0]: 52 bits -> byte 25 upper nibble, bytes 26..31 */
        {
            __m512i v0 = _mm512_set_epi64(
                (long long)src[7].n[0], (long long)src[6].n[0],
                (long long)src[5].n[0], (long long)src[4].n[0],
                (long long)src[3].n[0], (long long)src[2].n[0],
                (long long)src[1].n[0], (long long)src[0].n[0]);
            _mm512_store_si512((__m512i *)buf, v0);
            for (int j = 0; j < 8; j++) {
                uint64_t w = buf[j];
                dst[j][25] |= (uint8_t)((w >> 48) & 0x0F); /* OR upper nibble */
                dst[j][26] = (uint8_t)(w >> 40);
                dst[j][27] = (uint8_t)(w >> 32);
                dst[j][28] = (uint8_t)(w >> 24);
                dst[j][29] = (uint8_t)(w >> 16);
                dst[j][30] = (uint8_t)(w >> 8);
                dst[j][31] = (uint8_t)(w);
            }
        }
    }
}

/*
 * fe_to_sha256_words_16way: Convert SoA 52-bit limbs directly to SHA-256 big-endian SoA words.
 *
 * 5x52-bit limb layout:
 *   value = n[0] + n[1]*2^52 + n[2]*2^104 + n[3]*2^156 + n[4]*2^208
 *
 * 8x32-bit big-endian words (W[0] = most significant):
 *   W[0]: bits[255:224] = (n[4] >> 16)
 *   W[1]: bits[223:192] = (n[3] >> 36) | ((n[4] & 0xFFFF) << 16)
 *   W[2]: bits[191:160] = (n[3] >> 4) & 0xFFFFFFFF
 *   W[3]: bits[159:128] = (n[2] >> 24) | ((n[3] & 0xF) << 28)
 *   W[4]: bits[127:96]  = (n[1] >> 44) | ((n[2] & 0xFFFFFF) << 8)
 *   W[5]: bits[95:64]   = (n[1] >> 12) & 0xFFFFFFFF
 *   W[6]: bits[63:32]   = (n[0] >> 32) | ((n[1] & 0xFFF) << 20)
 *   W[7]: bits[31:0]    = n[0] & 0xFFFFFFFF
 *
 * Input:  fe->n[5][2] (each limb has lo/hi __m512i, 8 x 64-bit lanes each)
 * Output: w[8] (__m512i, 16 x 32-bit lanes each)
 *
 * Lane mapping: fe->n[i][0]'s 8 64-bit lanes produce the lower 8 32-bit lanes of w[j],
 *               fe->n[i][1]'s 8 64-bit lanes produce the upper 8 32-bit lanes of w[j].
 */
__attribute__((target("avx512f")))
void fe_to_sha256_words_16way(const secp256k1_fe_16x *fe, __m512i w[8])
{
    const __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFFLL);
    const __m512i mask12 = _mm512_set1_epi64(0xFFFLL);
    const __m512i mask24 = _mm512_set1_epi64(0xFFFFFFLL);
    const __m512i mask16 = _mm512_set1_epi64(0xFFFFLL);
    const __m512i mask4  = _mm512_set1_epi64(0xFLL);

    __m256i lo_words[8], hi_words[8];

    for (int half = 0; half < 2; half++) {
        __m512i n0 = fe->n[0][half];
        __m512i n1 = fe->n[1][half];
        __m512i n2 = fe->n[2][half];
        __m512i n3 = fe->n[3][half];
        __m512i n4 = fe->n[4][half];

        /* Compute 8 big-endian words (64-bit precision, truncated to 32-bit later) */
        __m512i w7_64 = _mm512_and_si512(n0, mask32);
        __m512i w6_64 = _mm512_or_si512(
            _mm512_srli_epi64(n0, 32),
            _mm512_slli_epi64(_mm512_and_si512(n1, mask12), 20));
        __m512i w5_64 = _mm512_and_si512(_mm512_srli_epi64(n1, 12), mask32);
        __m512i w4_64 = _mm512_or_si512(
            _mm512_srli_epi64(n1, 44),
            _mm512_slli_epi64(_mm512_and_si512(n2, mask24), 8));
        __m512i w3_64 = _mm512_or_si512(
            _mm512_srli_epi64(n2, 24),
            _mm512_slli_epi64(_mm512_and_si512(n3, mask4), 28));
        __m512i w2_64 = _mm512_and_si512(_mm512_srli_epi64(n3, 4), mask32);
        __m512i w1_64 = _mm512_or_si512(
            _mm512_srli_epi64(n3, 36),
            _mm512_slli_epi64(_mm512_and_si512(n4, mask16), 16));
        __m512i w0_64 = _mm512_and_si512(_mm512_srli_epi64(n4, 16), mask32);

        /* Pack 8x64-bit -> 8x32-bit: _mm512_cvtepi64_epi32 truncates 8 64-bit lanes
         * to __m256i (8 x 32-bit lanes) */
        __m256i h[8];
        h[0] = _mm512_cvtepi64_epi32(w0_64);
        h[1] = _mm512_cvtepi64_epi32(w1_64);
        h[2] = _mm512_cvtepi64_epi32(w2_64);
        h[3] = _mm512_cvtepi64_epi32(w3_64);
        h[4] = _mm512_cvtepi64_epi32(w4_64);
        h[5] = _mm512_cvtepi64_epi32(w5_64);
        h[6] = _mm512_cvtepi64_epi32(w6_64);
        h[7] = _mm512_cvtepi64_epi32(w7_64);

        if (half == 0) {
            for (int i = 0; i < 8; i++)
                lo_words[i] = h[i];
        } else {
            for (int i = 0; i < 8; i++)
                hi_words[i] = h[i];
        }
    }

    /* Merge lo/hi halves: lo -> lower 256 bits, hi -> upper 256 bits -> __m512i (16x32-bit) */
    for (int i = 0; i < 8; i++) {
        w[i] = _mm512_inserti64x4(_mm512_castsi256_si512(lo_words[i]), hi_words[i], 1);
    }
}

/*
 * keygen_ge_to_pubkey_bytes_16way: batch convert 16 affine points to pubkey bytes.
 *
 * Replaces 16 individual keygen_ge_to_pubkey_bytes calls with a single batch call.
 * Uses fe_get_b32_16way for vectorized field element -> byte conversion.
 * x_bytes are computed once and shared by both compressed and uncompressed paths.
 *
 * Parameters:
 *   ge              : array of 16 normalized affine points (infinity points are skipped)
 *   compressed_out  : array of 16 compressed pubkey buffers (33+ bytes each), or NULL to skip
 *   uncompressed_out: array of 16 uncompressed pubkey buffers (65+ bytes each), or NULL to skip
 */
void keygen_ge_to_pubkey_bytes_16way(const secp256k1_ge ge[16],
                                     uint8_t *compressed_out[16],
                                     uint8_t *uncompressed_out[16])
{
    /* Gather x coordinates (and y if uncompressed needed) */
    secp256k1_fe x_arr[16];
    for (int i = 0; i < 16; i++)
        x_arr[i] = ge[i].x;

    uint8_t x_bytes[16][32];
    fe_get_b32_16way(x_bytes, x_arr);

    if (compressed_out != NULL) {
        for (int i = 0; i < 16; i++) {
            if (compressed_out[i] == NULL)
                continue;
            compressed_out[i][0] = secp256k1_fe_is_odd(&ge[i].y) ? 0x03 : 0x02;
            memcpy(compressed_out[i] + 1, x_bytes[i], 32);
        }
    }

    if (uncompressed_out != NULL) {
        secp256k1_fe y_arr[16];
        for (int i = 0; i < 16; i++)
            y_arr[i] = ge[i].y;

        uint8_t y_bytes[16][32];
        fe_get_b32_16way(y_bytes, y_arr);

        for (int i = 0; i < 16; i++) {
            if (uncompressed_out[i] == NULL)
                continue;
            uncompressed_out[i][0] = 0x04;
            memcpy(uncompressed_out[i] + 1, x_bytes[i], 32);
            memcpy(uncompressed_out[i] + 1 + 32, y_bytes[i], 32);
        }
    }
}

#endif /* __AVX512IFMA__ */

