/*
 * gpu/gpu_secp256k1.cu
 *
 * GPU-side secp256k1 elliptic curve batch public key generation:
 *   - Finite field arithmetic (256-bit modular arithmetic, secp256k1 prime p)
 *   - Jacobian coordinate point addition (point add G)
 *   - Montgomery batch inversion (batch Z coordinate normalization)
 *   - Each CUDA thread handles one independent chain, incrementally deriving from base private key
 *
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
 *
 * Design notes:
 *   - secp256k1 prime p = 2^256 - 2^32 - 977
 *   - Uses 4x64-bit limbs to represent 256-bit integers
 *   - All operations performed in GPU registers, no global memory random access
 */

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gpu_interface.h"
#include "../keylog.h"

/* secp256k1 Curve Parameters */

/*
 * Prime p = 2^256 - 2^32 - 977
 * Represented as little-endian 64-bit limbs: p[0] is the lowest limb
 */
__device__ __constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

/* Group order n */
__device__ __constant__ uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

/* Generator G affine coordinates (little-endian 64-bit limbs) */
__device__ __constant__ uint64_t SECP256K1_GX[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

__device__ __constant__ uint64_t SECP256K1_GY[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};


/* 256-bit integer, little-endian 64-bit limbs */
typedef struct { uint64_t d[4]; } fe256;


/* Addition: r = a + b mod p */
__device__ __forceinline__ fe256 fe_add(const fe256 &a, const fe256 &b)
{
    fe256 r;
    uint64_t carry = 0;
    uint64_t t;

    /* 128-bit addition unrolled */
    t = a.d[0] + b.d[0];
    carry = (t < a.d[0]) ? 1 : 0;
    r.d[0] = t;

    t = a.d[1] + b.d[1] + carry;
    carry = (t < a.d[1] || (carry && t == a.d[1])) ? 1 : 0;
    r.d[1] = t;

    t = a.d[2] + b.d[2] + carry;
    carry = (t < a.d[2] || (carry && t == a.d[2])) ? 1 : 0;
    r.d[2] = t;

    t = a.d[3] + b.d[3] + carry;
    carry = (t < a.d[3] || (carry && t == a.d[3])) ? 1 : 0;
    r.d[3] = t;

    /* If overflow (carry=1) or r >= p, subtract p */
    /* secp256k1 p special structure: p = 2^256 - c, c = 2^32 + 977 = 0x1000003D1 */
    /* Condition: carry=1 means r >= 2^256 >= p; or r >= p without carry */
    /* Check r >= p: since p = 2^256 - c, r >= p iff r + c >= 2^256 */

    uint64_t c = 0x1000003D1ULL;
    /* Try r + c; if it overflows 256 bits (carry out), then r >= p, subtract p */
    uint64_t s0 = r.d[0] + c;
    uint64_t sc = (s0 < r.d[0]) ? 1 : 0;
    uint64_t s1 = r.d[1] + sc; sc = (s1 < r.d[1]) ? 1 : 0;
    uint64_t s2 = r.d[2] + sc; sc = (s2 < r.d[2]) ? 1 : 0;
    uint64_t s3 = r.d[3] + sc; sc = (s3 < r.d[3]) ? 1 : 0;
    /* sc=1 means r+c overflowed 256 bits, i.e. r >= p */
    if (carry || sc) {
        /* r -= p: equivalent to r += c (discard 2^256 overflow) */
        r.d[0] = s0; r.d[1] = s1; r.d[2] = s2; r.d[3] = s3;
    }

    return r;
}

/* Subtraction: r = a - b mod p */
__device__ __forceinline__ fe256 fe_sub(const fe256 &a, const fe256 &b)
{
    fe256 r;
    uint64_t borrow = 0;
    uint64_t t;

    t = a.d[0] - b.d[0];
    borrow = (a.d[0] < b.d[0]) ? 1 : 0;
    r.d[0] = t;

    /* Use separate borrow detection to avoid b.d[i] + borrow overflow when b.d[i] == 0xFFFFFFFFFFFFFFFF */
    t = a.d[1] - b.d[1] - borrow;
    borrow = (a.d[1] < b.d[1]) || (borrow && a.d[1] == b.d[1]) ? 1 : 0;
    r.d[1] = t;

    t = a.d[2] - b.d[2] - borrow;
    borrow = (a.d[2] < b.d[2]) || (borrow && a.d[2] == b.d[2]) ? 1 : 0;
    r.d[2] = t;

    t = a.d[3] - b.d[3] - borrow;
    borrow = (a.d[3] < b.d[3]) || (borrow && a.d[3] == b.d[3]) ? 1 : 0;
    r.d[3] = t;

    /* If underflow (borrow=1), add p */
    if (borrow) {
        /* r + p: since p = 2^256 - c (c = 0x1000003D1), adding p is equivalent to
         * subtracting c from the 256-bit value (the 2^256 overflow is discarded) */
        uint64_t c = 0x1000003D1ULL;
        uint64_t old0 = r.d[0];
        r.d[0] -= c;
        borrow = (old0 < c) ? 1 : 0;
        uint64_t old1 = r.d[1];
        r.d[1] -= borrow;
        borrow = (old1 < borrow) ? 1 : 0;
        uint64_t old2 = r.d[2];
        r.d[2] -= borrow;
        borrow = (old2 < borrow) ? 1 : 0;
        r.d[3] -= borrow;
    }
    return r;
}

/*
 * 128x128-bit multiplication, returns 256-bit result (low 128 bits in lo, high 128 bits in hi)
 * Uses CUDA built-in __umul64hi
 */
__device__ __forceinline__ void mul128(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi)
{
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

/*
 * P1 optimization: fe_mul using PTX inline assembly for 4x4 multiply core + secp256k1 fast reduction
 *
 * Rationale:
 *   Replace C-level __umul64hi + manual carry detection with PTX mad.lo/hi.cc.u64 + madc
 *   instruction chains, letting the compiler emit carry-flag-aware multiply-add sequences
 *   directly and eliminating redundant conditional branches and intermediate variables.
 *
 * PTX instruction reference:
 *   mul.lo.u64      rd, a, b       : rd = (a*b)[63:0]
 *   mul.hi.u64      rd, a, b       : rd = (a*b)[127:64]
 *   mad.lo.cc.u64   rd, a, b, c    : rd = (a*b)[63:0] + c,  sets CC.CF
 *   madc.lo.cc.u64  rd, a, b, c    : rd = (a*b)[63:0] + c + CC.CF, sets CC.CF
 *   madc.hi.cc.u64  rd, a, b, c    : rd = (a*b)[127:64] + c + CC.CF, sets CC.CF
 *   madc.hi.u64     rd, a, b, c    : rd = (a*b)[127:64] + c + CC.CF, no carry out (chain end)
 *   add.cc.u64      rd, a, b       : rd = a + b, sets CC.CF
 *   addc.cc.u64     rd, a, b       : rd = a + b + CC.CF, sets CC.CF
 *   addc.u64        rd, a, b       : rd = a + b + CC.CF, no carry out (chain end)
 *
 * The 4x4 multiply is unrolled into 16 partial products accumulated column-by-column
 * (along diagonals), propagating carries via CC.CF with no C-level conditionals.
 *
 * secp256k1 fast reduction: p = 2^256 - c, c = 2^32 + 977 = 0x1000003D1
 *   Round 1: fold t[4..7] * c back into t[0..3], yielding r[0..3] and overflow ov
 *   Round 2: fold ov * c back into r[0..1] (ov < 2^34, ov*c < 2^67, only low two limbs affected)
 *   Final conditional reduction: if r >= p (i.e. r + c overflows 256 bits), subtract p
 *                                (equivalent to keeping r + c and discarding the overflow)
 */
__device__ fe256 fe_mul(const fe256 &a, const fe256 &b)
{
    /*
     * Step 1: compute 512-bit product t[0..7] = a[0..3] * b[0..3]
     *
     * Row-major accumulation: row i adds a[i]*b[j] (lo and hi) into
     * t[i+j] and t[i+j+1].  After each madc.hi.cc, the carry is
     * propagated upward with addc.cc until it is absorbed (CF=0).
     * This guarantees correctness even when upper limbs are 0xffffffff...
     *
     * Instruction semantics:
     *   mad.lo.cc  rd,a,b,c : rd = lo(a*b)+c,        sets CC.CF (ignores old CF)
     *   madc.hi.cc rd,a,b,c : rd = hi(a*b)+c+CC.CF,  sets CC.CF
     *   addc.cc    rd,a,b   : rd = a+b+CC.CF,         sets CC.CF
     *   madc.hi.u64 rd,a,b,c: rd = hi(a*b)+c+CC.CF,  no CF out (chain end)
     */
    const uint64_t a0=a.d[0], a1=a.d[1], a2=a.d[2], a3=a.d[3];
    const uint64_t b0=b.d[0], b1=b.d[1], b2=b.d[2], b3=b.d[3];
    uint64_t t0, t1, t2, t3, t4, t5, t6, t7;

    /* ---- Row 0: a0 * b[0..3] -> t[0..4] ---- */
    asm(
        "mul.lo.u64       %0,  %5,  %9;        \n\t"   /* t0  = lo(a0*b0) */
        "mul.hi.u64       %1,  %5,  %9;        \n\t"   /* t1  = hi(a0*b0) */
        "mad.lo.cc.u64    %1,  %5, %10,   %1;  \n\t"   /* t1 += lo(a0*b1), CF */
        "madc.hi.cc.u64   %2,  %5, %10,    0;  \n\t"   /* t2  = hi(a0*b1)+CF, CF */
        "addc.cc.u64      %3,   0,   0;        \n\t"   /* t3  = CF, CF */
        "addc.u64         %4,   0,   0;        \n\t"   /* t4  = CF (chain end; t3<=1, no overflow) */
        "mad.lo.cc.u64    %2,  %5, %11,   %2;  \n\t"   /* t2 += lo(a0*b2), CF */
        "madc.hi.cc.u64   %3,  %5, %11,   %3;  \n\t"   /* t3 += hi(a0*b2)+CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t4 += CF (chain end; t4<=1, no overflow) */
        "mad.lo.cc.u64    %3,  %5, %12,   %3;  \n\t"   /* t3 += lo(a0*b3), CF */
        "madc.hi.u64      %4,  %5, %12,   %4;  \n\t"   /* t4 += hi(a0*b3)+CF (end) */
        : "=&l"(t0),"=&l"(t1),"=&l"(t2),"=&l"(t3),"=&l"(t4)
        : "l"(a0),"l"(a1),"l"(a2),"l"(a3),
          "l"(b0),"l"(b1),"l"(b2),"l"(b3)
    );
    t5 = 0; t6 = 0; t7 = 0;

    /* ---- Row 1: a1 * b[0..3] -> add into t[1..5] ---- */
    asm(
        /* hi(a1*b0) carry: propagate to t3, t4, t5 (t4 may overflow) */
        "mad.lo.cc.u64    %0,  %6,  %9,   %0;  \n\t"   /* t1 += lo(a1*b0), CF */
        "madc.hi.cc.u64   %1,  %6,  %9,   %1;  \n\t"   /* t2 += hi(a1*b0)+CF, CF */
        "addc.cc.u64      %2,  %2,   0;        \n\t"   /* t3 += CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t4 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t5 += CF (chain end) */
        /* hi(a1*b1) carry: propagate to t4, t5 */
        "mad.lo.cc.u64    %1,  %6, %10,   %1;  \n\t"   /* t2 += lo(a1*b1), CF */
        "madc.hi.cc.u64   %2,  %6, %10,   %2;  \n\t"   /* t3 += hi(a1*b1)+CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t4 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t5 += CF (chain end) */
        /* hi(a1*b2) carry: propagate to t5 */
        "mad.lo.cc.u64    %2,  %6, %11,   %2;  \n\t"   /* t3 += lo(a1*b2), CF */
        "madc.hi.cc.u64   %3,  %6, %11,   %3;  \n\t"   /* t4 += hi(a1*b2)+CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t5 += CF (chain end) */
        "mad.lo.cc.u64    %3,  %6, %12,   %3;  \n\t"   /* t4 += lo(a1*b3), CF */
        "madc.hi.u64      %4,  %6, %12,   %4;  \n\t"   /* t5 += hi(a1*b3)+CF (end) */
        : "+l"(t1),"+l"(t2),"+l"(t3),"+l"(t4),"+l"(t5)
        : "l"(a0),"l"(a1),"l"(a2),"l"(a3),
          "l"(b0),"l"(b1),"l"(b2),"l"(b3)
    );

    /* ---- Row 2: a2 * b[0..3] -> add into t[2..6] ---- */
    asm(
        /* hi(a2*b0) carry: propagate to t4, t5, t6, t7 (t6 may overflow) */
        "mad.lo.cc.u64    %0,  %7,  %9,   %0;  \n\t"   /* t2 += lo(a2*b0), CF */
        "madc.hi.cc.u64   %1,  %7,  %9,   %1;  \n\t"   /* t3 += hi(a2*b0)+CF, CF */
        "addc.cc.u64      %2,  %2,   0;        \n\t"   /* t4 += CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t5 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t6 += CF (chain end; t6=0, no overflow) */
        /* hi(a2*b1) carry: propagate to t5, t6 */
        "mad.lo.cc.u64    %1,  %7, %10,   %1;  \n\t"   /* t3 += lo(a2*b1), CF */
        "madc.hi.cc.u64   %2,  %7, %10,   %2;  \n\t"   /* t4 += hi(a2*b1)+CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t5 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t6 += CF (chain end) */
        /* hi(a2*b2) carry: propagate to t6 */
        "mad.lo.cc.u64    %2,  %7, %11,   %2;  \n\t"   /* t4 += lo(a2*b2), CF */
        "madc.hi.cc.u64   %3,  %7, %11,   %3;  \n\t"   /* t5 += hi(a2*b2)+CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t6 += CF (chain end) */
        "mad.lo.cc.u64    %3,  %7, %12,   %3;  \n\t"   /* t5 += lo(a2*b3), CF */
        "madc.hi.u64      %4,  %7, %12,   %4;  \n\t"   /* t6 += hi(a2*b3)+CF (end) */
        : "+l"(t2),"+l"(t3),"+l"(t4),"+l"(t5),"+l"(t6)
        : "l"(a0),"l"(a1),"l"(a2),"l"(a3),
          "l"(b0),"l"(b1),"l"(b2),"l"(b3)
    );

    /* ---- Row 3: a3 * b[0..3] -> add into t[3..7] ---- */
    asm(
        /* hi(a3*b0) carry: propagate to t5, t6, t7 */
        "mad.lo.cc.u64    %0,  %8,  %9,   %0;  \n\t"   /* t3 += lo(a3*b0), CF */
        "madc.hi.cc.u64   %1,  %8,  %9,   %1;  \n\t"   /* t4 += hi(a3*b0)+CF, CF */
        "addc.cc.u64      %2,  %2,   0;        \n\t"   /* t5 += CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t6 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t7 += CF (chain end) */
        /* hi(a3*b1) carry: propagate to t6, t7 */
        "mad.lo.cc.u64    %1,  %8, %10,   %1;  \n\t"   /* t4 += lo(a3*b1), CF */
        "madc.hi.cc.u64   %2,  %8, %10,   %2;  \n\t"   /* t5 += hi(a3*b1)+CF, CF */
        "addc.cc.u64      %3,  %3,   0;        \n\t"   /* t6 += CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t7 += CF (chain end) */
        /* hi(a3*b2) carry: propagate to t7 */
        "mad.lo.cc.u64    %2,  %8, %11,   %2;  \n\t"   /* t5 += lo(a3*b2), CF */
        "madc.hi.cc.u64   %3,  %8, %11,   %3;  \n\t"   /* t6 += hi(a3*b2)+CF, CF */
        "addc.u64         %4,  %4,   0;        \n\t"   /* t7 += CF (chain end) */
        "mad.lo.cc.u64    %3,  %8, %12,   %3;  \n\t"   /* t6 += lo(a3*b3), CF */
        "madc.hi.u64      %4,  %8, %12,   %4;  \n\t"   /* t7 += hi(a3*b3)+CF (end) */
        : "+l"(t3),"+l"(t4),"+l"(t5),"+l"(t6),"+l"(t7)
        : "l"(a0),"l"(a1),"l"(a2),"l"(a3),
          "l"(b0),"l"(b1),"l"(b2),"l"(b3)
    );

    /*
     * Step 2: secp256k1 fast reduction
     *   p = 2^256 - c, c = 2^32 + 977 = 0x1000003D1
     *   t mod p = t_lo + t_hi * c mod p
     *
     * Round 1 reduction: r[0..3] = t[0..3] + t[4..7] * c
     *   Each t[i]*c is 97 bits (t[i] 64-bit, c 33-bit); mul.lo/hi extract the low/high 64 bits.
     *   Accumulate column-by-column using CC.CF to propagate carries; final overflow ov < 2^34.
     *
     * Input constraints : %5~%8 = t0..t3, %9~%12 = t4..t7, %13 = c
     * Output constraints: %0~%3 = r0..r3, %4 = ov
     */
    const uint64_t C = 0x1000003D1ULL;
    uint64_t r0, r1, r2, r3, ov;

    asm(
        /* r0 = t0 + t4*c_lo */
        "mul.lo.u64      %0,  %9, %13;       \n\t"   /* tmp = t4*c lo */
        "add.cc.u64      %0,  %0,  %5;       \n\t"   /* r0 = tmp + t0, set CF */

        /* r1 = t1 + t4*c_hi + t5*c_lo + CF */
        "madc.hi.cc.u64  %1,  %9, %13,  %6;  \n\t"   /* r1 = t4*c hi + t1 + CF */
        "madc.lo.cc.u64  %1, %10, %13,  %1;  \n\t"   /* r1 += t5*c lo + CF */

        /* r2 = t2 + t5*c_hi + t6*c_lo + CF */
        "madc.hi.cc.u64  %2, %10, %13,  %7;  \n\t"   /* r2 = t5*c hi + t2 + CF */
        "madc.lo.cc.u64  %2, %11, %13,  %2;  \n\t"   /* r2 += t6*c lo + CF */

        /* r3 = t3 + t6*c_hi + t7*c_lo + CF */
        "madc.hi.cc.u64  %3, %11, %13,  %8;  \n\t"   /* r3 = t6*c hi + t3 + CF */
        "madc.lo.cc.u64  %3, %12, %13,  %3;  \n\t"   /* r3 += t7*c lo + CF */

        /* ov = t7*c_hi + CF (overflow, < 2^34) */
        "madc.hi.u64     %4, %12, %13,   0;  \n\t"   /* ov = t7*c hi + CF (chain end) */

        : "=&l"(r0), "=&l"(r1), "=&l"(r2), "=&l"(r3), "=&l"(ov)
        : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
          "l"(t4), "l"(t5), "l"(t6), "l"(t7), "l"(C)
    );

    /*
     * Round 2 reduction: fold overflow ov * c back into r[0..1]
     *   ov < 2^34, ov*c < 2^67, only r0 and r1 are affected
     */
    uint64_t ov_lo, ov_hi;
    asm(
        "mul.lo.u64  %0, %2, %3; \n\t"
        "mul.hi.u64  %1, %2, %3; \n\t"
        : "=&l"(ov_lo), "=&l"(ov_hi)
        : "l"(ov), "l"(C)
    );
    asm(
        "add.cc.u64  %0, %0, %2; \n\t"   /* r0 += ov_lo, set CF */
        "addc.u64    %1, %1, %3; \n\t"   /* r1 += ov_hi + CF (chain end) */
        : "+l"(r0), "+l"(r1)
        : "l"(ov_lo), "l"(ov_hi)
    );

    /*
     * Final conditional reduction: result is in [0, 2p); subtract p if r >= p
     *   r >= p  <=>  r + c overflows 256 bits (since p = 2^256 - c)
     */
    uint64_t s0, s1, s2, s3, sc;
    asm(
        "add.cc.u64   %0, %5, %9;  \n\t"   /* s0 = r0 + c, set CF */
        "addc.cc.u64  %1, %6,  0;  \n\t"   /* s1 = r1 + CF */
        "addc.cc.u64  %2, %7,  0;  \n\t"   /* s2 = r2 + CF */
        "addc.cc.u64  %3, %8,  0;  \n\t"   /* s3 = r3 + CF */
        "addc.u64     %4,  0,  0;  \n\t"   /* sc = CF (overflow flag, chain end) */
        : "=&l"(s0), "=&l"(s1), "=&l"(s2), "=&l"(s3), "=&l"(sc)
        : "l"(r0), "l"(r1), "l"(r2), "l"(r3), "l"(C)
    );
    if (sc) { r0 = s0; r1 = s1; r2 = s2; r3 = s3; }

    fe256 res;
    res.d[0] = r0; res.d[1] = r1; res.d[2] = r2; res.d[3] = r3;
    return res;
}

/*
 * H1 optimization: dedicated PTX squaring fe_sqr(a) = a^2 mod p
 *
 * Rationale:
 *   For squaring a = [a0, a1, a2, a3], the 4x4 product has 16 partial products,
 *   but only 10 are distinct (diagonal terms a_i*a_i appear once; off-diagonal
 *   terms a_i*a_j with i<j appear twice as a_i*a_j + a_j*a_i = 2*a_i*a_j).
 *   We compute the 10 unique products and double the off-diagonal ones via left-shift,
 *   saving ~6 mul.hi instructions compared to calling fe_mul(a, a).
 *
 * Column layout (t[k] = sum of a_i*a_j where i+j=k):
 *   t0 = a0*a0
 *   t1 = 2*a0*a1
 *   t2 = 2*a0*a2 + a1*a1
 *   t3 = 2*a0*a3 + 2*a1*a2
 *   t4 = 2*a1*a3 + a2*a2
 *   t5 = 2*a2*a3
 *   t6 = a3*a3   (no t7 carry needed; t6 holds the full high word)
 *   (t7 = 0 since a3*a3 fits in 128 bits and the sum is at most 512 bits)
 *
 * Strategy: compute off-diagonal products first (without the factor of 2),
 *   then double them using add.cc/addc chains, then add diagonal terms.
 *   This keeps the PTX carry chains short and avoids extra registers.
 */
__device__ fe256 fe_sqr(const fe256 &a)
{
    uint64_t t0, t1, t2, t3, t4, t5, t6, t7;

    asm(
        /* ---- diagonal t0 = a0*a0 ---- */
        "mul.lo.u64      %0,  %8,  %8;       \n\t"   /* t0 = a0*a0 lo */

        /* ---- off-diagonal column 1 (before doubling): od1 = a0*a1 ---- */
        "mul.lo.u64      %1,  %8,  %9;       \n\t"   /* t1 = a0*a1 lo */
        "mul.hi.u64      %2,  %8,  %9;       \n\t"   /* t2 = a0*a1 hi */
        "mul.hi.u64      %7,  %8,  %8;       \n\t"   /* tmp(t7) = a0*a0 hi (pre-compute before doubling) */

        /* double od1: t1=2*(a0*a1 lo), t2=2*(a0*a1 hi)+CF, t3=CF */
        "add.cc.u64      %1,  %1,  %1;       \n\t"   /* t1 = 2*(a0*a1 lo), set CF */
        "addc.cc.u64     %2,  %2,  %2;       \n\t"   /* t2 = 2*(a0*a1 hi) + CF, set CF */
        "addc.u64        %3,   0,   0;       \n\t"   /* t3 = CF (chain end, captures t2 overflow) */

        /* add diagonal a0*a0 hi into t1 */
        "add.cc.u64      %1,  %1,  %7;       \n\t"   /* t1 += a0*a0 hi, set CF */
        "addc.cc.u64     %2,  %2,   0;       \n\t"   /* t2 += CF, set CF */
        "addc.u64        %3,  %3,   0;       \n\t"   /* t3 += CF (chain end) */

        /* ---- off-diagonal column 2: od2 = a0*a2 ---- */
        /* first pass */
        "mad.lo.cc.u64   %2,  %8, %10,  %2;  \n\t"   /* t2 += a0*a2 lo, set CF */
        "madc.hi.cc.u64  %3,  %8, %10,  %3;  \n\t"   /* t3 += a0*a2 hi + CF, set CF */
        "addc.u64        %4,   0,   0;       \n\t"   /* t4 = CF (chain end) */

        /* second pass (double) */
        "mad.lo.cc.u64   %2,  %8, %10,  %2;  \n\t"   /* t2 += a0*a2 lo again, set CF */
        "madc.hi.cc.u64  %3,  %8, %10,  %3;  \n\t"   /* t3 += a0*a2 hi + CF, set CF */
        "addc.u64        %4,  %4,   0;       \n\t"   /* t4 += CF (chain end) */

        /* add diagonal a1*a1 lo into t2 */
        "mad.lo.cc.u64   %2,  %9,  %9,  %2;  \n\t"   /* t2 += a1*a1 lo, set CF */
        "addc.cc.u64     %3,  %3,   0;       \n\t"   /* t3 += CF, set CF */
        "addc.u64        %4,  %4,   0;       \n\t"   /* t4 += CF (chain end) */

        /* ---- off-diagonal column 3: od3 = a0*a3 + a1*a2 ---- */
        /* first pass: a0*a3 */
        "mad.lo.cc.u64   %3,  %8, %11,  %3;  \n\t"   /* t3 += a0*a3 lo, set CF */
        "madc.hi.cc.u64  %4,  %8, %11,  %4;  \n\t"   /* t4 += a0*a3 hi + CF, set CF */
        "addc.u64        %5,   0,   0;       \n\t"   /* t5 = CF (chain end) */
        /* first pass: a1*a2 */
        "mad.lo.cc.u64   %3,  %9, %10,  %3;  \n\t"   /* t3 += a1*a2 lo, set CF */
        "madc.hi.cc.u64  %4,  %9, %10,  %4;  \n\t"   /* t4 += a1*a2 hi + CF, set CF */
        "addc.u64        %5,  %5,   0;       \n\t"   /* t5 += CF (chain end) */

        /* second pass (double): a0*a3 */
        "mad.lo.cc.u64   %3,  %8, %11,  %3;  \n\t"   /* t3 += a0*a3 lo again, set CF */
        "madc.hi.cc.u64  %4,  %8, %11,  %4;  \n\t"   /* t4 += a0*a3 hi + CF, set CF */
        "addc.cc.u64     %5,  %5,   0;       \n\t"   /* t5 += CF, set CF */
        "addc.u64        %6,   0,   0;       \n\t"   /* t6 = CF (chain end) */
        /* second pass: a1*a2 */
        "mad.lo.cc.u64   %3,  %9, %10,  %3;  \n\t"   /* t3 += a1*a2 lo again, set CF */
        "madc.hi.cc.u64  %4,  %9, %10,  %4;  \n\t"   /* t4 += a1*a2 hi + CF, set CF */
        "addc.cc.u64     %5,  %5,   0;       \n\t"   /* t5 += CF, set CF */
        "addc.u64        %6,  %6,   0;       \n\t"   /* t6 += CF (chain end) */

        /* add diagonal a1*a1 hi into t3 */
        "mad.hi.cc.u64   %3,  %9,  %9,  %3;  \n\t"   /* t3 += a1*a1 hi, set CF */
        "addc.cc.u64     %4,  %4,   0;       \n\t"   /* t4 += CF, set CF */
        "addc.u64        %5,  %5,   0;       \n\t"   /* t5 += CF (chain end) */

        /* ---- off-diagonal column 4: od4 = a1*a3 ---- */
        /* first pass */
        "mad.lo.cc.u64   %4,  %9, %11,  %4;  \n\t"   /* t4 += a1*a3 lo, set CF */
        "addc.cc.u64     %5,  %5,   0;       \n\t"   /* t5 += CF, set CF (must clear CC.CF before madc.hi.cc) */
        "madc.hi.cc.u64  %5,  %9, %11,  %5;  \n\t"   /* t5 += a1*a3 hi + CF(=0), set CF */
        "addc.u64        %6,  %6,   0;       \n\t"   /* t6 += CF (chain end) */

        /* second pass (double) */
        "mad.lo.cc.u64   %4,  %9, %11,  %4;  \n\t"   /* t4 += a1*a3 lo again, set CF */
        "addc.cc.u64     %5,  %5,   0;       \n\t"   /* t5 += CF, set CF (must clear CC.CF before madc.hi.cc) */
        "madc.hi.cc.u64  %5,  %9, %11,  %5;  \n\t"   /* t5 += a1*a3 hi + CF(=0), set CF */
        "addc.u64        %6,  %6,   0;       \n\t"   /* t6 += CF (chain end) */

        /* add diagonal a2*a2 lo into t4 */
        "mad.lo.cc.u64   %4, %10, %10,  %4;  \n\t"   /* t4 += a2*a2 lo, set CF */
        "addc.u64        %5,  %5,   0;       \n\t"   /* t5 += CF (chain end) */

        /* ---- off-diagonal column 5: od5 = a2*a3 ---- */
        /* first pass */
        "mad.lo.cc.u64   %5, %10, %11,  %5;  \n\t"   /* t5 += a2*a3 lo, set CF */
        "addc.cc.u64     %6,  %6,   0;       \n\t"   /* t6 += CF, set CF (must clear CC.CF before madc.hi.cc) */
        "madc.hi.cc.u64  %6, %10, %11,  %6;  \n\t"   /* t6 += a2*a3 hi + CF(=0), set CF */
        "addc.u64        %7,   0,   0;       \n\t"   /* t7 = CF (chain end) */

        /* second pass (double) */
        "mad.lo.cc.u64   %5, %10, %11,  %5;  \n\t"   /* t5 += a2*a3 lo again, set CF */
        "addc.cc.u64     %6,  %6,   0;       \n\t"   /* t6 += CF, set CF (must clear CC.CF before madc.hi.cc) */
        "madc.hi.cc.u64  %6, %10, %11,  %6;  \n\t"   /* t6 += a2*a3 hi + CF(=0), set CF */
        "addc.u64        %7,  %7,   0;       \n\t"   /* t7 += CF (chain end) */

        /* add diagonal a2*a2 hi into t5 */
        "mad.hi.cc.u64   %5, %10, %10,  %5;  \n\t"   /* t5 += a2*a2 hi, set CF */
        "addc.u64        %6,  %6,   0;       \n\t"   /* t6 += CF (chain end) */

        /* ---- diagonal t6 += a3*a3 lo ---- */
        "mad.lo.cc.u64   %6, %11, %11,  %6;  \n\t"   /* t6 += a3*a3 lo, set CF */
        "addc.cc.u64     %7,  %7,   0;       \n\t"   /* t7 += CF, clear CC.CF (must clear before madc.hi) */

        /* ---- t7 += a3*a3 hi ---- */
        "madc.hi.u64     %7, %11, %11,  %7;  \n\t"   /* t7 += a3*a3 hi + CC.CF(=0) */

        : "=&l"(t0), "=&l"(t1), "=&l"(t2), "=&l"(t3),
          "=&l"(t4), "=&l"(t5), "=&l"(t6), "=&l"(t7)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3])
    );

    /* Reduction: identical to fe_mul (p = 2^256 - c, c = 0x1000003D1) */
    const uint64_t C = 0x1000003D1ULL;
    uint64_t r0, r1, r2, r3, ov;

    asm(
        "mul.lo.u64      %0,  %9, %13;       \n\t"
        "add.cc.u64      %0,  %0,  %5;       \n\t"
        "madc.hi.cc.u64  %1,  %9, %13,  %6;  \n\t"
        "madc.lo.cc.u64  %1, %10, %13,  %1;  \n\t"
        "madc.hi.cc.u64  %2, %10, %13,  %7;  \n\t"
        "madc.lo.cc.u64  %2, %11, %13,  %2;  \n\t"
        "madc.hi.cc.u64  %3, %11, %13,  %8;  \n\t"
        "madc.lo.cc.u64  %3, %12, %13,  %3;  \n\t"
        "madc.hi.u64     %4, %12, %13,   0;  \n\t"
        : "=&l"(r0), "=&l"(r1), "=&l"(r2), "=&l"(r3), "=&l"(ov)
        : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
          "l"(t4), "l"(t5), "l"(t6), "l"(t7), "l"(C)
    );

    uint64_t ov_lo, ov_hi;
    asm(
        "mul.lo.u64  %0, %2, %3; \n\t"
        "mul.hi.u64  %1, %2, %3; \n\t"
        : "=&l"(ov_lo), "=&l"(ov_hi)
        : "l"(ov), "l"(C)
    );
    asm(
        "add.cc.u64  %0, %0, %2; \n\t"
        "addc.u64    %1, %1, %3; \n\t"
        : "+l"(r0), "+l"(r1)
        : "l"(ov_lo), "l"(ov_hi)
    );

    uint64_t s0, s1, s2, s3, sc;
    asm(
        "add.cc.u64   %0, %5, %9;  \n\t"
        "addc.cc.u64  %1, %6,  0;  \n\t"
        "addc.cc.u64  %2, %7,  0;  \n\t"
        "addc.cc.u64  %3, %8,  0;  \n\t"
        "addc.u64     %4,  0,  0;  \n\t"
        : "=&l"(s0), "=&l"(s1), "=&l"(s2), "=&l"(s3), "=&l"(sc)
        : "l"(r0), "l"(r1), "l"(r2), "l"(r3), "l"(C)
    );
    if (sc) { r0 = s0; r1 = s1; r2 = s2; r3 = s3; }

    fe256 res;
    res.d[0] = r0; res.d[1] = r1; res.d[2] = r2; res.d[3] = r3;
    return res;
}

/*
 * Modular inverse: r = a^(-1) mod p
 * Uses Fermat's little theorem: a^(-1) = a^(p-2) mod p
 * p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
 * Uses addition chain to optimize exponentiation
 */
__device__ fe256 fe_inv(const fe256 &a)
{
    /* Use standard secp256k1 modular inverse addition chain */
    fe256 x2  = fe_mul(fe_sqr(a), a);
    fe256 x3  = fe_mul(fe_sqr(x2), a);
    fe256 x6  = fe_mul(fe_sqr(fe_sqr(fe_sqr(x3))), x3);
    fe256 x9  = fe_mul(fe_sqr(fe_sqr(fe_sqr(x6))), x3);
    fe256 x11 = fe_mul(fe_sqr(fe_sqr(x9)), x2);
    fe256 x22 = fe_mul(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(x11))))))))))), x11);
    fe256 x44 = fe_mul(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(x22)))))))))))))))))))))), x22);
    fe256 x88 = fe_mul(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(x44)))))))))))))))))))))))))))))))))))))))))))), x44);
    fe256 x176 = fe_mul(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(x88)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))), x88);
    fe256 x220 = fe_mul(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(fe_sqr(
                    fe_sqr(fe_sqr(fe_sqr(fe_sqr(x176)))))))))))))))))))))))))))))))))))))))))))), x44);
    fe256 x223 = fe_mul(fe_sqr(fe_sqr(fe_sqr(x220))), x3);

    /* Final: a^(p-2) using correct addition chain */
    /* Verified: sqr^23, *x22, sqr^5, *a, sqr^3, *x2, sqr^2, *a */
    fe256 t1 = x223;
    for (int i = 0; i < 23; i++)
        t1 = fe_sqr(t1);
    t1 = fe_mul(t1, x22);
    for (int i = 0; i < 5; i++)
        t1 = fe_sqr(t1);
    t1 = fe_mul(t1, a);
    for (int i = 0; i < 3; i++)
        t1 = fe_sqr(t1);
    t1 = fe_mul(t1, x2);
    t1 = fe_sqr(t1);
    t1 = fe_sqr(t1);
    t1 = fe_mul(t1, a);
    return t1;
}

/* Check if fe256 is zero */
__device__ __forceinline__ int fe_is_zero(const fe256 &a)
{
    return (a.d[0] == 0 && a.d[1] == 0 && a.d[2] == 0 && a.d[3] == 0);
}

/*
 * Montgomery Batch Inversion: compute modular inverse of n field elements in batch
 * Algorithm:
 *   Forward accumulation: acc[i] = in[0] * in[1] * ... * in[i]
 *   Call fe_inv once on acc[n-1]
 *   Backward substitution: out[i] = acc[i-1] * (acc[n-1]^-1 * in[n-1] * ... * in[i+1])
 *
 * Parameters:
 *   out       : output array, out[i] = in[i]^-1 mod p (out[i]=0 if in[i]==0)
 *   in        : input array
 *   n         : number of elements
 *   valid_out : if non-NULL, set *valid_out to the index of the first zero element
 *               (truncation marker); caller should initialize *valid_out to n
 *               (meaning all valid) before calling
 *
 * Note: if in[i] == 0, then out[i] = 0 and *valid_out is set to i (truncated)
 */
__device__ void fe_batch_inv(
    fe256       * __restrict__ out,
    const fe256 * __restrict__ in,
    int n,
    int         * __restrict__ valid_out)
{
    if (n <= 0) return;

    /* Forward accumulation: acc[i] = in[0] * ... * in[i] */
    /* Reuse out[] as temporary storage for acc[] to save registers */
    if (fe_is_zero(in[0])) {
        if (valid_out != NULL)
            *valid_out = 0;
        return;
    }
    out[0] = in[0];
    int first_zero = n;  /* index of first zero element; n means no zero found */

    for (int i = 1; i < n; i++) {
        if (fe_is_zero(in[i])) {
            /* Zero element encountered: truncate batch processing */
            first_zero = i;
            break;
        }
        out[i] = fe_mul(out[i-1], in[i]);
    }

    /* Number of valid elements */
    int valid_n = first_zero;

    if (valid_out != NULL) {
        *valid_out = valid_n;
    }

    if (valid_n == 0) {
        /* First element is zero */
        return;
    }

    /* Invert the last accumulated product: inv_acc = (in[0]*...*in[valid_n-1])^-1 */
    fe256 inv_acc = fe_inv(out[valid_n - 1]);

    /* Backward substitution:
     *   running = inv_acc
     *   out[i] = running * out[i-1]  (i.e. in[i]^-1)
     *   running = running * in[i]    (update to (in[0]*...*in[i-1])^-1)
     */
    for (int i = valid_n - 1; i >= 1; i--) {
        /* out[i] = inv_acc * out[i-1] = in[i]^-1 */
        fe256 tmp = fe_mul(inv_acc, out[i-1]);
        /* inv_acc = inv_acc * in[i] = (in[0]*...*in[i-1])^-1 */
        inv_acc = fe_mul(inv_acc, in[i]);
        out[i] = tmp;
    }
    /* out[0] = inv_acc = in[0]^-1 */
    out[0] = inv_acc;

    /* Zero out entries beyond valid range */
    for (int i = valid_n; i < n; i++) {
        out[i].d[0] = 0; out[i].d[1] = 0;
        out[i].d[2] = 0; out[i].d[3] = 0;
    }
}

/*
 * Strided batch inversion: same algorithm as fe_batch_inv but accesses elements
 * at stride intervals (step-major global memory layout).
 *   out[i*stride] = in[i*stride]^-1  for i in [0, n)
 * This avoids thread-local temporary arrays, eliminating stack frame spilling.
 */
__device__ void fe_batch_inv_strided(
    fe256       * __restrict__ out,   /* base pointer, stride = num_chains */
    const fe256 * __restrict__ in,    /* base pointer, stride = num_chains */
    int n,
    int stride,
    int         * __restrict__ valid_out)
{
    if (n <= 0) return;

    if (fe_is_zero(in[0])) {
        if (valid_out != NULL)
            *valid_out = 0;
        return;
    }
    out[0] = in[0];
    int first_zero = n;

    for (int i = 1; i < n; i++) {
        if (fe_is_zero(in[i * stride])) {
            first_zero = i;
            break;
        }
        out[i * stride] = fe_mul(out[(i-1) * stride], in[i * stride]);
    }

    int valid_n = first_zero;
    if (valid_out != NULL)
        *valid_out = valid_n;

    if (valid_n == 0)
        return;

    fe256 inv_acc = fe_inv(out[(valid_n - 1) * stride]);

    for (int i = valid_n - 1; i >= 1; i--) {
        fe256 tmp = fe_mul(inv_acc, out[(i-1) * stride]);
        inv_acc   = fe_mul(inv_acc, in[i * stride]);
        out[i * stride] = tmp;
    }
    out[0] = inv_acc;

    for (int i = valid_n; i < n; i++) {
        out[i * stride].d[0] = 0; out[i * stride].d[1] = 0;
        out[i * stride].d[2] = 0; out[i * stride].d[3] = 0;
    }
}

/* Jacobian Coordinate Point Operations */

typedef struct {
    fe256 x, y, z;
    int infinity;   /* 1 = point at infinity */
} jac_point;

typedef struct {
    fe256 x, y;
} aff_point;

/* Convert 32-byte big-endian byte array to fe256 (little-endian limbs) */
__device__ __forceinline__ fe256 fe_from_bytes(const uint8_t *b)
{
    fe256 r;
    r.d[3] = ((uint64_t)b[0]  << 56) | ((uint64_t)b[1]  << 48) |
             ((uint64_t)b[2]  << 40) | ((uint64_t)b[3]  << 32) |
             ((uint64_t)b[4]  << 24) | ((uint64_t)b[5]  << 16) |
             ((uint64_t)b[6]  <<  8) | ((uint64_t)b[7]);
    r.d[2] = ((uint64_t)b[8]  << 56) | ((uint64_t)b[9]  << 48) |
             ((uint64_t)b[10] << 40) | ((uint64_t)b[11] << 32) |
             ((uint64_t)b[12] << 24) | ((uint64_t)b[13] << 16) |
             ((uint64_t)b[14] <<  8) | ((uint64_t)b[15]);
    r.d[1] = ((uint64_t)b[16] << 56) | ((uint64_t)b[17] << 48) |
             ((uint64_t)b[18] << 40) | ((uint64_t)b[19] << 32) |
             ((uint64_t)b[20] << 24) | ((uint64_t)b[21] << 16) |
             ((uint64_t)b[22] <<  8) | ((uint64_t)b[23]);
    r.d[0] = ((uint64_t)b[24] << 56) | ((uint64_t)b[25] << 48) |
             ((uint64_t)b[26] << 40) | ((uint64_t)b[27] << 32) |
             ((uint64_t)b[28] << 24) | ((uint64_t)b[29] << 16) |
             ((uint64_t)b[30] <<  8) | ((uint64_t)b[31]);
    return r;
}

/* Convert fe256 to 32-byte big-endian byte array */
__device__ __forceinline__ void fe_to_bytes(const fe256 &a, uint8_t *b)
{
    b[0]  = (uint8_t)(a.d[3] >> 56); b[1]  = (uint8_t)(a.d[3] >> 48);
    b[2]  = (uint8_t)(a.d[3] >> 40); b[3]  = (uint8_t)(a.d[3] >> 32);
    b[4]  = (uint8_t)(a.d[3] >> 24); b[5]  = (uint8_t)(a.d[3] >> 16);
    b[6]  = (uint8_t)(a.d[3] >>  8); b[7]  = (uint8_t)(a.d[3]);
    b[8]  = (uint8_t)(a.d[2] >> 56); b[9]  = (uint8_t)(a.d[2] >> 48);
    b[10] = (uint8_t)(a.d[2] >> 40); b[11] = (uint8_t)(a.d[2] >> 32);
    b[12] = (uint8_t)(a.d[2] >> 24); b[13] = (uint8_t)(a.d[2] >> 16);
    b[14] = (uint8_t)(a.d[2] >>  8); b[15] = (uint8_t)(a.d[2]);
    b[16] = (uint8_t)(a.d[1] >> 56); b[17] = (uint8_t)(a.d[1] >> 48);
    b[18] = (uint8_t)(a.d[1] >> 40); b[19] = (uint8_t)(a.d[1] >> 32);
    b[20] = (uint8_t)(a.d[1] >> 24); b[21] = (uint8_t)(a.d[1] >> 16);
    b[22] = (uint8_t)(a.d[1] >>  8); b[23] = (uint8_t)(a.d[1]);
    b[24] = (uint8_t)(a.d[0] >> 56); b[25] = (uint8_t)(a.d[0] >> 48);
    b[26] = (uint8_t)(a.d[0] >> 40); b[27] = (uint8_t)(a.d[0] >> 32);
    b[28] = (uint8_t)(a.d[0] >> 24); b[29] = (uint8_t)(a.d[0] >> 16);
    b[30] = (uint8_t)(a.d[0] >>  8); b[31] = (uint8_t)(a.d[0]);
}

/* Check if fe256 is odd (lowest bit) */
__device__ __forceinline__ int fe_is_odd(const fe256 &a)
{
    return (int)(a.d[0] & 1);
}

/*
 * Jacobian point addition: R = P + G (G in affine coordinates)
 * Formula (mixed addition, G.z = 1):
 *   U1 = P.x
 *   U2 = G.x * P.z^2
 *   S1 = P.y
 *   S2 = G.y * P.z^3
 *   H  = U2 - U1
 *   R  = S2 - S1
 *   R.x = R^2 - H^3 - 2*U1*H^2
 *   R.y = R*(U1*H^2 - R.x) - S1*H^3
 *   R.z = P.z * H
 */
__device__ jac_point jac_add_affine_G(const jac_point &P)
{
    if (P.infinity) {
        /* P = O, result is G */
        jac_point R;
        /* SECP256K1_GX/GY are stored as little-endian uint64_t limbs, copy directly */
        R.x.d[0] = SECP256K1_GX[0]; R.x.d[1] = SECP256K1_GX[1];
        R.x.d[2] = SECP256K1_GX[2]; R.x.d[3] = SECP256K1_GX[3];
        R.y.d[0] = SECP256K1_GY[0]; R.y.d[1] = SECP256K1_GY[1];
        R.y.d[2] = SECP256K1_GY[2]; R.y.d[3] = SECP256K1_GY[3];
        R.z.d[0] = 1; R.z.d[1] = 0; R.z.d[2] = 0; R.z.d[3] = 0;
        R.infinity = 0;
        return R;
    }

    fe256 Gx, Gy;
    Gx.d[0] = SECP256K1_GX[0]; Gx.d[1] = SECP256K1_GX[1];
    Gx.d[2] = SECP256K1_GX[2]; Gx.d[3] = SECP256K1_GX[3];
    Gy.d[0] = SECP256K1_GY[0]; Gy.d[1] = SECP256K1_GY[1];
    Gy.d[2] = SECP256K1_GY[2]; Gy.d[3] = SECP256K1_GY[3];

    fe256 Z2 = fe_sqr(P.z);
    fe256 Z3 = fe_mul(Z2, P.z);
    fe256 U2 = fe_mul(Gx, Z2);
    fe256 S2 = fe_mul(Gy, Z3);

    fe256 H = fe_sub(U2, P.x);
    fe256 R = fe_sub(S2, P.y);

    if (fe_is_zero(H)) {
        if (fe_is_zero(R)) {
            /* P == G: perform point doubling (2G) using standard EFD Jacobian doubling */
            /* secp256k1 (a=0): https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l */
            /* A = X1^2, B = Y1^2, C = B^2 */
            /* D = 2*((X1+B)^2 - A - C) = 4*X1*Y1^2 */
            /* E = 3*A, F = E^2 */
            /* X3 = F - 2*D */
            /* Y3 = E*(D - X3) - 8*C */
            /* Z3 = 2*Y1*Z1 */
            fe256 dbl_A  = fe_sqr(P.x);                          /* X1^2 */
            fe256 dbl_B  = fe_sqr(P.y);                          /* Y1^2 */
            fe256 dbl_C  = fe_sqr(dbl_B);                        /* Y1^4 */
            fe256 dbl_XB = fe_add(P.x, dbl_B);                   /* X1+B */
            fe256 dbl_D  = fe_add(fe_sub(fe_sqr(dbl_XB), dbl_A), /* (X1+B)^2-A-C */
                                  fe_sub(fe_sqr(dbl_XB), dbl_A));
            dbl_D = fe_sub(dbl_D, fe_add(dbl_C, dbl_C));         /* 2*((X1+B)^2-A-C) */
            fe256 dbl_E  = fe_add(fe_add(dbl_A, dbl_A), dbl_A);  /* 3*A */
            fe256 dbl_F  = fe_sqr(dbl_E);                        /* E^2 */
            fe256 Rx_dbl = fe_sub(dbl_F, fe_add(dbl_D, dbl_D));  /* F - 2*D */
            fe256 Ry_dbl = fe_sub(
                fe_mul(dbl_E, fe_sub(dbl_D, Rx_dbl)),             /* E*(D-X3) */
                fe_add(fe_add(fe_add(dbl_C, dbl_C), fe_add(dbl_C, dbl_C)),
                       fe_add(fe_add(dbl_C, dbl_C), fe_add(dbl_C, dbl_C))) /* 8*C */
            );
            fe256 Rz_dbl = fe_add(fe_mul(P.y, P.z), fe_mul(P.y, P.z)); /* 2*Y1*Z1 */
            jac_point res;
            res.x = Rx_dbl; res.y = Ry_dbl; res.z = Rz_dbl;
            res.infinity = 0;
            return res;
        } else {
            /* P == -G: P + G = point at infinity */
            jac_point res;
            res.infinity = 1;
            return res;
        }
    }

    fe256 H2 = fe_sqr(H);
    fe256 H3 = fe_mul(H2, H);
    fe256 U1H2 = fe_mul(P.x, H2);

    fe256 R2 = fe_sqr(R);
    fe256 Rx = fe_sub(fe_sub(R2, H3), fe_add(U1H2, U1H2));
    fe256 Ry = fe_sub(fe_mul(R, fe_sub(U1H2, Rx)), fe_mul(P.y, H3));
    fe256 Rz = fe_mul(P.z, H);

    jac_point res;
    res.x = Rx; res.y = Ry; res.z = Rz;
    res.infinity = 0;
    return res;
}

/*
 * General Jacobian + Affine mixed point addition: R = P (Jacobian) + Q (Affine)
 * Formula (mixed addition, Q.z = 1):
 *   U1 = P.x
 *   U2 = Q.x * P.z^2
 *   S1 = P.y
 *   S2 = Q.y * P.z^3
 *   H  = U2 - U1
 *   R  = S2 - S1
 *   X3 = R^2 - H^3 - 2*U1*H^2
 *   Y3 = R*(U1*H^2 - X3) - S1*H^3
 *   Z3 = P.z * H
 */
__device__ jac_point jac_add_affine(const jac_point &P, const aff_point &Q)
{
    if (P.infinity) {
        jac_point R;
        R.x = Q.x; R.y = Q.y;
        R.z.d[0] = 1; R.z.d[1] = 0; R.z.d[2] = 0; R.z.d[3] = 0;
        R.infinity = 0;
        return R;
    }

    fe256 Z2 = fe_sqr(P.z);
    fe256 Z3 = fe_mul(Z2, P.z);
    fe256 U2 = fe_mul(Q.x, Z2);
    fe256 S2 = fe_mul(Q.y, Z3);

    fe256 H = fe_sub(U2, P.x);
    fe256 R = fe_sub(S2, P.y);

    if (fe_is_zero(H)) {
        if (fe_is_zero(R)) {
            /* P == Q: perform point doubling using standard EFD Jacobian doubling */
            /* secp256k1 (a=0): https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l */
            /* A = X1^2, B = Y1^2, C = B^2 */
            /* D = 2*((X1+B)^2 - A - C) = 4*X1*Y1^2 */
            /* E = 3*A, F = E^2 */
            /* X3 = F - 2*D, Y3 = E*(D-X3) - 8*C, Z3 = 2*Y1*Z1 */
            fe256 dbl_A  = fe_sqr(P.x);
            fe256 dbl_B  = fe_sqr(P.y);
            fe256 dbl_C  = fe_sqr(dbl_B);
            fe256 dbl_XB = fe_add(P.x, dbl_B);
            fe256 dbl_D  = fe_add(fe_sub(fe_sqr(dbl_XB), dbl_A),
                                  fe_sub(fe_sqr(dbl_XB), dbl_A));
            dbl_D = fe_sub(dbl_D, fe_add(dbl_C, dbl_C));
            fe256 dbl_E  = fe_add(fe_add(dbl_A, dbl_A), dbl_A);
            fe256 dbl_F  = fe_sqr(dbl_E);
            fe256 Rx_dbl = fe_sub(dbl_F, fe_add(dbl_D, dbl_D));
            fe256 Ry_dbl = fe_sub(
                fe_mul(dbl_E, fe_sub(dbl_D, Rx_dbl)),
                fe_add(fe_add(fe_add(dbl_C, dbl_C), fe_add(dbl_C, dbl_C)),
                       fe_add(fe_add(dbl_C, dbl_C), fe_add(dbl_C, dbl_C))) /* 8*C */
            );
            fe256 Rz_dbl = fe_add(fe_mul(P.y, P.z), fe_mul(P.y, P.z)); /* 2*Y1*Z1 */
            jac_point res;
            res.x = Rx_dbl; res.y = Ry_dbl; res.z = Rz_dbl;
            res.infinity = 0;
            return res;
        } else {
            /* P == -Q: result is point at infinity */
            jac_point res;
            res.infinity = 1;
            return res;
        }
    }

    fe256 H2 = fe_sqr(H);
    fe256 H3 = fe_mul(H2, H);
    fe256 U1H2 = fe_mul(P.x, H2);

    fe256 R2 = fe_sqr(R);
    fe256 Rx = fe_sub(fe_sub(R2, H3), fe_add(U1H2, U1H2));
    fe256 Ry = fe_sub(fe_mul(R, fe_sub(U1H2, Rx)), fe_mul(P.y, H3));
    fe256 Rz = fe_mul(P.z, H);

    jac_point res;
    res.x = Rx; res.y = Ry; res.z = Rz;
    res.infinity = 0;
    return res;
}

/*
 * Jacobian point doubling: R = 2*P (secp256k1, a=0)
 * Uses EFD dbl-2009-l formula:
 *   A = X1^2, B = Y1^2, C = B^2
 *   D = 2*((X1+B)^2 - A - C)
 *   E = 3*A, F = E^2
 *   X3 = F - 2*D
 *   Y3 = E*(D - X3) - 8*C
 *   Z3 = 2*Y1*Z1
 * No fe_inv required.
 */
__device__ jac_point jac_double(const jac_point &P)
{
    if (P.infinity) {
        jac_point R;
        R.infinity = 1;
        return R;
    }

    fe256 A  = fe_sqr(P.x);                              /* X1^2 */
    fe256 B  = fe_sqr(P.y);                              /* Y1^2 */
    fe256 C  = fe_sqr(B);                                /* B^2 = Y1^4 */
    fe256 XB = fe_add(P.x, B);                           /* X1+B */
    /* D = 2*((X1+B)^2 - A - C) */
    fe256 D  = fe_sub(fe_sub(fe_sqr(XB), A), C);        /* (X1+B)^2 - A - C */
    D = fe_add(D, D);                                    /* 2*(...) */
    fe256 E  = fe_add(fe_add(A, A), A);                  /* 3*A */
    fe256 F  = fe_sqr(E);                                /* E^2 */
    fe256 X3 = fe_sub(F, fe_add(D, D));                  /* F - 2*D */
    /* Y3 = E*(D - X3) - 8*C */
    fe256 C2 = fe_add(C, C);                             /* 2*C */
    fe256 C4 = fe_add(C2, C2);                           /* 4*C */
    fe256 C8 = fe_add(C4, C4);                           /* 8*C */
    fe256 Y3 = fe_sub(fe_mul(E, fe_sub(D, X3)), C8);
    /* Z3 = 2*Y1*Z1 */
    fe256 Z3 = fe_add(fe_mul(P.y, P.z), fe_mul(P.y, P.z));

    jac_point R;
    R.x = X3; R.y = Y3; R.z = Z3;
    R.infinity = 0;
    return R;
}

/*
 * Convert Jacobian point to affine coordinates: (X:Y:Z) -> (X/Z^2, Y/Z^3)
 * Calls fe_inv once.
 */
__device__ aff_point jac_to_affine(const jac_point &P)
{
    aff_point R;
    fe256 z_inv  = fe_inv(P.z);
    fe256 z_inv2 = fe_sqr(z_inv);           /* Z^-2 */
    fe256 z_inv3 = fe_mul(z_inv2, z_inv);   /* Z^-3 */
    R.x = fe_mul(P.x, z_inv2);
    R.y = fe_mul(P.y, z_inv3);
    return R;
}

/*
 * General Jacobian + Jacobian point addition: R = P + Q
 * Uses EFD add-2007-bl formula:
 *   U1 = X1*Z2^2, U2 = X2*Z1^2
 *   S1 = Y1*Z2^3, S2 = Y2*Z1^3
 *   H  = U2 - U1, R  = S2 - S1
 *   X3 = R^2 - H^3 - 2*U1*H^2
 *   Y3 = R*(U1*H^2 - X3) - S1*H^3
 *   Z3 = H*Z1*Z2
 * Handles infinity and point-doubling edge cases.
 */
__device__ jac_point jac_add(const jac_point &P, const jac_point &Q)
{
    if (P.infinity) return Q;
    if (Q.infinity) return P;

    fe256 Z1_2 = fe_sqr(P.z);
    fe256 Z2_2 = fe_sqr(Q.z);
    fe256 U1   = fe_mul(P.x, Z2_2);
    fe256 U2   = fe_mul(Q.x, Z1_2);
    fe256 S1   = fe_mul(P.y, fe_mul(Z2_2, Q.z));   /* Y1*Z2^3 */
    fe256 S2   = fe_mul(Q.y, fe_mul(Z1_2, P.z));   /* Y2*Z1^3 */

    fe256 H = fe_sub(U2, U1);
    fe256 Rv = fe_sub(S2, S1);

    if (fe_is_zero(H)) {
        if (fe_is_zero(Rv)) {
            /* P == Q: use dedicated doubling */
            return jac_double(P);
        } else {
            /* P == -Q: result is point at infinity */
            jac_point res;
            res.infinity = 1;
            return res;
        }
    }

    fe256 H2   = fe_sqr(H);
    fe256 H3   = fe_mul(H2, H);
    fe256 U1H2 = fe_mul(U1, H2);

    fe256 Rv2  = fe_sqr(Rv);
    fe256 X3   = fe_sub(fe_sub(Rv2, H3), fe_add(U1H2, U1H2));
    fe256 Y3   = fe_sub(fe_mul(Rv, fe_sub(U1H2, X3)), fe_mul(S1, H3));
    fe256 Z3   = fe_mul(fe_mul(H, P.z), Q.z);

    jac_point res;
    res.x = X3; res.y = Y3; res.z = Z3;
    res.infinity = 0;
    return res;
}

/*
 * Compute scalar * G (Jacobian coordinates) from 32-byte private key (big-endian)
 * Uses LSB-first double-and-add algorithm.
 *
 * Optimization (P0): addend is now tracked in Jacobian coordinates using jac_double,
 * which eliminates all fe_inv calls inside the loop (was 256 calls, now 0).
 * The caller is responsible for converting the returned Jacobian result to affine
 * via jac_to_affine (one fe_inv call total).
 */
__device__ jac_point scalar_mult_G(const uint8_t *privkey)
{
    jac_point R;
    R.infinity = 1;

    /* Current multiple of G in Jacobian coordinates, starting from 1*G (z=1) */
    jac_point addend;
    addend.x.d[0] = SECP256K1_GX[0]; addend.x.d[1] = SECP256K1_GX[1];
    addend.x.d[2] = SECP256K1_GX[2]; addend.x.d[3] = SECP256K1_GX[3];
    addend.y.d[0] = SECP256K1_GY[0]; addend.y.d[1] = SECP256K1_GY[1];
    addend.y.d[2] = SECP256K1_GY[2]; addend.y.d[3] = SECP256K1_GY[3];
    addend.z.d[0] = 1; addend.z.d[1] = 0; addend.z.d[2] = 0; addend.z.d[3] = 0;
    addend.infinity = 0;

    /* Scan 256-bit scalar from lowest bit (LSB-first double-and-add) */
    /* privkey is big-endian: privkey[0] is MSB, privkey[31] is LSB */
    for (int byte_idx = 31; byte_idx >= 0; byte_idx--) {
        uint8_t b = privkey[byte_idx];
        for (int bit = 0; bit < 8; bit++) {
            if (b & (1 << bit)) {
                /* R = R + addend (Jacobian + Jacobian addition, no fe_inv) */
                R = jac_add(R, addend);
            }
            /* addend = 2 * addend (Jacobian doubling, no fe_inv) */
            addend = jac_double(addend);
        }
    }
    return R;
}

/* GPU Kernel */

/*
 * Global device buffers (allocated by gpu_secp256k1_alloc)
 * Layout:
 *   d_base_privkeys : [num_chains * 32] bytes, base private key for each chain
 *   d_aff_x        : [num_chains * GPU_CHAIN_STEPS * 32] bytes, affine X coordinates
 *   d_aff_y        : [num_chains * GPU_CHAIN_STEPS * 32] bytes, affine Y coordinates
 *   d_valid        : [num_chains] int, chain validity flags
 *   d_jac_X/Y    : [num_chains * GPU_CHAIN_STEPS * 32] bytes, Jacobian X/Y coords scratch
 *
 * jac_Z and z_inv are thread-local arrays (8 KB each) for cache-friendly batch inversion.
 * eliminating the 16 KB stack frame per thread and raising SM occupancy.
 */
static uint8_t *d_base_privkeys = NULL;
static uint8_t *d_aff_x        = NULL;
static uint8_t *d_aff_y        = NULL;
static int     *d_valid         = NULL;
static uint8_t *d_jac_X        = NULL;   /* Jacobian X scratch: num_chains * GPU_CHAIN_STEPS * 32 */
static uint8_t *d_jac_Y        = NULL;   /* Jacobian Y scratch */
static uint8_t *d_jac_Z        = NULL;   /* Jacobian Z scratch: step-major layout */
static uint8_t *d_z_inv        = NULL;   /* batch-inversion result scratch: step-major layout */
static int      g_num_chains    = 0;

/*
 * Kernel 1: batch public key generation
 *
 * Scheme B: Global Batch Inversion
 *
 * Optimization:
 *   The previous tiled scheme called fe_batch_inv once per TILE_SIZE steps,
 *   resulting in steps/TILE_SIZE fe_inv calls total.
 *   This scheme collects all Jacobian Z coordinates for all steps first, then
 *   calls fe_batch_inv exactly once, reducing fe_inv calls by ~97%.
 *
 * Trade-off:
 *   Requires storing steps Jacobian X/Y/Z coordinates in thread-local arrays,
 *   increasing local memory spilling. However, fe_inv compute cost far outweighs
 *   the extra local memory traffic.
 *
 * Memory estimate (steps=GPU_CHAIN_STEPS=256):
 *   jac_X/Y/Z: 3 * 256 * 32 = 24 KB/thread (local memory spilling)
 *   z_inv:     1 * 256 * 32 =  8 KB/thread
 *   Total ~32 KB/thread; local memory increases but fe_inv savings dominate.
 *
 * Input:
 *   base_privkeys : [num_chains * 32] bytes, base private key for each chain (big-endian)
 * Output:
 *   aff_x, aff_y  : [num_chains * steps * 32] bytes, affine coordinates
 *   valid         : [num_chains] int, 0=invalid, >0=valid step count
 */

__global__ __launch_bounds__(128, 4) void kernel_gen_pubkeys(
    const uint8_t * __restrict__ base_privkeys,
    uint8_t       * __restrict__ aff_x,
    uint8_t       * __restrict__ aff_y,
    int           * __restrict__ valid,
    uint8_t       * __restrict__ g_jac_X,   /* step-major scratch: steps*num_chains*32 bytes */
    uint8_t       * __restrict__ g_jac_Y,
    uint8_t       * __restrict__ g_jac_Z,   /* step-major scratch for Z coordinates */
    uint8_t       * __restrict__ g_z_inv,   /* step-major scratch for batch-inv results */
    int num_chains,
    int steps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chains)
        return;

    const uint8_t *privkey = base_privkeys + tid * 32;

    /* Reject zero private key (invalid) */
    int all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (privkey[i] != 0) { all_zero = 0; break; }
    }
    if (all_zero) {
        valid[tid] = 0;
        return;
    }

    /* Compute base public key: P0 = privkey * G */
    jac_point P_jac = scalar_mult_G(privkey);
    if (P_jac.infinity) {
        valid[tid] = 0;
        return;
    }
    /* Convert to affine then re-wrap as Jacobian (z=1) for a normalized starting point */
    aff_point P0_aff = jac_to_affine(P_jac);
    jac_point P;
    P.x = P0_aff.x; P.y = P0_aff.y;
    P.z.d[0] = 1; P.z.d[1] = 0; P.z.d[2] = 0; P.z.d[3] = 0;
    P.infinity = 0;

    /* Current scalar value, used for overflow detection */
    fe256 cur_scalar = fe_from_bytes(privkey);

    /* ---- Scheme B: global batch inversion ---- */
    /* jac_X / jac_Y: step-major layout [step * num_chains + tid]
     *   → same-step threads are adjacent in memory → coalesced 128-byte reads/writes
     * jac_Z / z_inv: thread-local arrays (8 KB each)
     *   → fe_batch_inv requires contiguous input; local array avoids stride cache miss */
    fe256 * __restrict__ jac_X_base = (fe256 *)g_jac_X;  /* step-major: [step*num_chains+tid] */
    fe256 * __restrict__ jac_Y_base = (fe256 *)g_jac_Y;
    fe256 * __restrict__ jac_Z_base = (fe256 *)g_jac_Z;  /* step-major: [step*num_chains+tid] */
    fe256 * __restrict__ z_inv_base = (fe256 *)g_z_inv;  /* step-major: [step*num_chains+tid] */

    /* ---- Phase 1: full Jacobian traversal, collect all X/Y/Z ---- */
    int actual_steps = steps;   /* effective step count, truncated on scalar overflow */
    for (int i = 0; i < steps; i++) {
        /* step-major write: coalesced across warp (32 threads write 32 consecutive fe256) */
        jac_X_base[i * num_chains + tid] = P.x;
        jac_Y_base[i * num_chains + tid] = P.y;
        jac_Z_base[i * num_chains + tid] = P.z;

        if (i < steps - 1) {
            /* P = P + G */
            P = jac_add_affine_G(P);

            /* Scalar overflow check (extremely rare: scalar + 1 wraps to 0) */
            cur_scalar.d[0]++;
            if (cur_scalar.d[0] == 0) {
                cur_scalar.d[1]++;
                if (cur_scalar.d[1] == 0) {
                    cur_scalar.d[2]++;
                    if (cur_scalar.d[2] == 0) {
                        cur_scalar.d[3]++;
                    }
                }
            }
            if (fe_is_zero(cur_scalar)) {
                actual_steps = i + 1;   /* truncate to current step */
                break;
            }
        }
    }

    /* ---- Phase 2: single global batch inversion (strided, no thread-local temp arrays) ---- */
    int batch_valid = actual_steps;
    fe_batch_inv_strided(
        z_inv_base + tid,          /* out base: tid-th column, stride = num_chains */
        jac_Z_base + tid,          /* in  base: tid-th column, stride = num_chains */
        actual_steps,
        num_chains,
        &batch_valid);
    if (batch_valid < actual_steps) {
        actual_steps = batch_valid;
    }

    /* ---- Phase 3: compute affine coordinates and write to output buffer ---- */
    for (int i = 0; i < actual_steps; i++) {
        fe256 zi  = z_inv_base[i * num_chains + tid];
        fe256 zi2 = fe_sqr(zi);
        fe256 zi3 = fe_mul(zi2, zi);
        /* step-major read: coalesced across warp */
        fe256 ax  = fe_mul(jac_X_base[i * num_chains + tid], zi2);
        fe256 ay  = fe_mul(jac_Y_base[i * num_chains + tid], zi3);

        /* output also step-major for coalesced writes */
        int out_idx = i * num_chains + tid;
        fe_to_bytes(ax, aff_x + out_idx * 32);
        fe_to_bytes(ay, aff_y + out_idx * 32);
    }

    valid[tid] = actual_steps;
}

/*
 * Allocate GPU-side secp256k1 work buffers
 * num_chains : number of chains (= GPU_BATCH_SIZE)
 * Return value: 0 success, -1 failure
 */
int gpu_secp256k1_alloc(int num_chains)
{
    g_num_chains = num_chains;
    size_t privkey_size = (size_t)num_chains * 32;
    size_t coord_size = (size_t)num_chains * GPU_CHAIN_STEPS * 32;

    if (cudaMalloc(&d_base_privkeys, privkey_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_aff_x, coord_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_aff_y, coord_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_valid, (size_t)num_chains * sizeof(int)) != cudaSuccess)
        goto fail;
    /* Jacobian X/Y/Z and z_inv scratch buffers: step-major layout [step * num_chains + tid] */
    if (cudaMalloc(&d_jac_X, coord_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_jac_Y, coord_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_jac_Z, coord_size) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_z_inv, coord_size) != cudaSuccess)
        goto fail;

    keylog_info("[GPU] secp256k1 buffers allocated: %d chains x %d steps",
                num_chains, GPU_CHAIN_STEPS);
    return 0;

fail:
    keylog_error("[GPU] secp256k1 buffer allocation failed");
    return -1;
}

/*
 * Free GPU-side secp256k1 work buffers
 */
void gpu_secp256k1_free(void)
{
    if (d_base_privkeys) {
        cudaFree(d_base_privkeys);
        d_base_privkeys = NULL;
    }
    if (d_aff_x) {
        cudaFree(d_aff_x);
        d_aff_x = NULL;
    }
    if (d_aff_y) {
        cudaFree(d_aff_y);
        d_aff_y = NULL;
    }
    if (d_valid) {
        cudaFree(d_valid);
        d_valid = NULL;
    }
    if (d_jac_X) {
        cudaFree(d_jac_X);
        d_jac_X = NULL;
    }
    if (d_jac_Y) {
        cudaFree(d_jac_Y);
        d_jac_Y = NULL;
    }
    if (d_jac_Z) {
        cudaFree(d_jac_Z);
        d_jac_Z = NULL;
    }
    if (d_z_inv) {
        cudaFree(d_z_inv);
        d_z_inv = NULL;
    }
}

/*
 * Execute batch public key generation kernel
 * h_base_privkeys : CPU-side base private key array (num_chains * 32 bytes)
 * stream          : CUDA stream for async execution
 * Return value: 0 success, -1 failure
 */
int gpu_secp256k1_run(const uint8_t *h_base_privkeys, cudaStream_t stream)
{
    /* Async transfer base private keys to GPU */
    size_t privkey_size = (size_t)g_num_chains * 32;
    if (cudaMemcpyAsync(d_base_privkeys, h_base_privkeys, privkey_size,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        keylog_error("[GPU] Base private key async transfer failed");
        return -1;
    }

    /* Configure kernel: 128 threads per block (matches __launch_bounds__ for better occupancy) */
    int block_size = 128;
    int grid_size  = (g_num_chains + block_size - 1) / block_size;

    kernel_gen_pubkeys<<<grid_size, block_size, 0, stream>>>(
        d_base_privkeys, d_aff_x, d_aff_y, d_valid,
        d_jac_X, d_jac_Y, d_jac_Z, d_z_inv,
        g_num_chains, GPU_CHAIN_STEPS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        keylog_error("[GPU] kernel_gen_pubkeys execution failed: %s", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/* Return device-side affine coordinate buffer pointers (for use by gpu_hash160.cu) */
const uint8_t *gpu_secp256k1_get_aff_x(void)
{
    return d_aff_x;
}

const uint8_t *gpu_secp256k1_get_aff_y(void)
{
    return d_aff_y;
}

const int *gpu_secp256k1_get_valid(void)
{
    return d_valid;
}

int gpu_secp256k1_get_num_chains(void)
{
    return g_num_chains;
}

/* Return device-side base private key buffer pointer (for use by gpu_hashtable.cu kernel) */
const uint8_t *gpu_secp256k1_get_base_privkeys(void)
{
    return d_base_privkeys;
}

