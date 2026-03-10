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

/* Modular multiplication: r = a * b mod p (fast reduction using secp256k1 p special structure) */
__device__ fe256 fe_mul(const fe256 &a, const fe256 &b)
{
    /* Compute 512-bit product t[0..7] */
    uint64_t t[8] = {0};
    uint64_t lo, hi, carry;

    /* Unrolled 4x4 multiplication with correct carry propagation */
    /* For each partial product a[i]*b[j]: add lo to t[i+j], add hi to t[i+j+1],
     * then propagate all carries upward. */
#define MULADD(i, j) \
    do { \
        mul128(a.d[i], b.d[j], &lo, &hi); \
        /* Step 1: t[i+j] += lo, record carry c1 */ \
        uint64_t _old0 = t[i+j]; \
        t[i+j] += lo; \
        uint64_t _c1 = (t[i+j] < _old0) ? 1 : 0; \
        /* Step 2: t[i+j+1] += hi, record carry c2 */ \
        uint64_t _old1 = t[i+j+1]; \
        t[i+j+1] += hi; \
        uint64_t _c2 = (t[i+j+1] < _old1) ? 1 : 0; \
        /* Step 3: t[i+j+1] += c1 (carry from step 1), record carry c3 */ \
        if (_c1) { \
            t[i+j+1]++; \
            _c2 += (t[i+j+1] == 0) ? 1 : 0; \
        } \
        /* Step 4: propagate _c2 upward from t[i+j+2] */ \
        carry = _c2; \
        for (int _k = i+j+2; _k < 8 && carry; _k++) { \
            uint64_t _old_k = t[_k]; \
            t[_k] += carry; \
            carry = (t[_k] < _old_k) ? 1 : 0; \
        } \
    } while(0)

    MULADD(0,0); MULADD(0,1); MULADD(0,2); MULADD(0,3);
    MULADD(1,0); MULADD(1,1); MULADD(1,2); MULADD(1,3);
    MULADD(2,0); MULADD(2,1); MULADD(2,2); MULADD(2,3);
    MULADD(3,0); MULADD(3,1); MULADD(3,2); MULADD(3,3);
#undef MULADD

    /*
     * secp256k1 fast reduction: p = 2^256 - c, c = 2^32 + 977 = 0x1000003D1
     * For 512-bit product t = t_lo + t_hi * 2^256:
     *   t mod p = t_lo + t_hi * c mod p
     * Two iterations reduce result to [0, 2p)
     */
    uint64_t c = 0x1000003D1ULL;

    /* First reduction: fold t[4..7] back into t[0..3] */
    uint64_t r[4];
    uint64_t acc = 0;

    /* Add low 256 bits of t_hi * c to t_lo */
    /* Accumulate t[4]*c, t[5]*c, t[6]*c, t[7]*c in sequence */
    uint64_t h0, h1, h2, h3;
    uint64_t l0, l1, l2, l3;
    mul128(t[4], c, &l0, &h0);
    mul128(t[5], c, &l1, &h1);
    mul128(t[6], c, &l2, &h2);
    mul128(t[7], c, &l3, &h3);

    /* Accumulate into t[0..3] with correct multi-operand carry detection */
    acc = t[0] + l0;
    carry = (acc < t[0]) ? 1 : 0;
    r[0] = acc;

    /* t[1] + l1 + h0 + carry: add step by step to detect each carry */
    acc = t[1] + l1;
    uint64_t c1 = (acc < t[1]) ? 1 : 0;
    acc += h0;
    c1 += (acc < h0) ? 1 : 0;
    acc += carry;
    c1 += (acc < carry) ? 1 : 0;
    r[1] = acc;
    carry = c1;

    acc = t[2] + l2;
    uint64_t c2 = (acc < t[2]) ? 1 : 0;
    acc += h1;
    c2 += (acc < h1) ? 1 : 0;
    acc += carry;
    c2 += (acc < carry) ? 1 : 0;
    r[2] = acc;
    carry = c2;

    acc = t[3] + l3;
    uint64_t c3 = (acc < t[3]) ? 1 : 0;
    acc += h2;
    c3 += (acc < h2) ? 1 : 0;
    acc += carry;
    c3 += (acc < carry) ? 1 : 0;
    r[3] = acc;
    carry = c3;

    /* Handle overflow: fold (carry + h3) multiplied by c */
    /* carry and h3 are both small (< 4), so their sum won't overflow uint64_t */
    uint64_t overflow = carry + h3;
    uint64_t extra_lo, extra_hi;
    mul128(overflow, c, &extra_lo, &extra_hi);

    acc = r[0] + extra_lo;
    carry = (acc < r[0]) ? 1 : 0;
    r[0] = acc;
    /* Step-by-step to correctly detect carry from multi-operand addition */
    acc = r[1] + extra_hi;
    uint64_t c4 = (acc < r[1]) ? 1 : 0;
    acc += carry;
    c4 += (acc < carry) ? 1 : 0;
    r[1] = acc;
    carry = c4;
    r[2] += carry;
    carry = (r[2] < carry) ? 1 : 0;
    r[3] += carry;

    /* Final conditional reduction: result may be in [0, 2p), reduce to [0, p) */
    /* Check r >= p: r >= p iff r + c overflows 256 bits */
    /* Note: c = 0x1000003D1ULL is already declared above, reuse it */
    uint64_t s0 = r[0] + c; uint64_t sc = (s0 < r[0]) ? 1 : 0;
    uint64_t s1 = r[1] + sc; sc = (s1 < r[1]) ? 1 : 0;
    uint64_t s2 = r[2] + sc; sc = (s2 < r[2]) ? 1 : 0;
    uint64_t s3 = r[3] + sc; sc = (s3 < r[3]) ? 1 : 0;
    if (sc) {
        r[0] = s0;
        r[1] = s1;
        r[2] = s2;
        r[3] = s3;
    }

    fe256 res;
    res.d[0] = r[0]; res.d[1] = r[1]; res.d[2] = r[2]; res.d[3] = r[3];
    return res;
}

/* Squaring: r = a^2 mod p */
__device__ __forceinline__ fe256 fe_sqr(const fe256 &a)
{
    return fe_mul(a, a);
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
 * Compute scalar * G (Jacobian coordinates) from 32-byte private key (big-endian)
 * Uses LSB-first double-and-add algorithm
 */
__device__ jac_point scalar_mult_G(const uint8_t *privkey)
{
    jac_point R;
    R.infinity = 1;

    fe256 Gx, Gy;
    Gx.d[0] = SECP256K1_GX[0]; Gx.d[1] = SECP256K1_GX[1];
    Gx.d[2] = SECP256K1_GX[2]; Gx.d[3] = SECP256K1_GX[3];
    Gy.d[0] = SECP256K1_GY[0]; Gy.d[1] = SECP256K1_GY[1];
    Gy.d[2] = SECP256K1_GY[2]; Gy.d[3] = SECP256K1_GY[3];

    /* Current multiple of G in affine coordinates, starting from 1*G */
    aff_point addend;
    addend.x = Gx;
    addend.y = Gy;

    /* Scan 256-bit scalar from lowest bit (LSB-first double-and-add) */
    /* privkey is big-endian: privkey[0] is MSB, privkey[31] is LSB */
    for (int byte_idx = 31; byte_idx >= 0; byte_idx--) {
        uint8_t b = privkey[byte_idx];
        for (int bit = 0; bit < 8; bit++) {
            if (b & (1 << bit)) {
                /* R = R + addend (general Jacobian + Affine mixed addition) */
                R = jac_add_affine(R, addend);
            }
            /* addend = 2 * addend (affine point doubling) */
            /* lambda = 3*x^2 / (2*y), x' = lambda^2 - 2*x, y' = lambda*(x - x') - y */
            fe256 x2 = fe_sqr(addend.x);
            fe256 lam_num = fe_add(fe_add(x2, x2), x2);  /* 3*x^2 */
            fe256 lam_den = fe_add(addend.y, addend.y);   /* 2*y */
            fe256 lam = fe_mul(lam_num, fe_inv(lam_den));
            fe256 lam2 = fe_sqr(lam);
            fe256 nx = fe_sub(fe_sub(lam2, addend.x), addend.x);
            fe256 ny = fe_sub(fe_mul(lam, fe_sub(addend.x, nx)), addend.y);
            addend.x = nx;
            addend.y = ny;
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
 */
static uint8_t *d_base_privkeys = NULL;
static uint8_t *d_aff_x        = NULL;
static uint8_t *d_aff_y        = NULL;
static int     *d_valid         = NULL;
static int      g_num_chains    = 0;

/*
 * Kernel 1: batch public key generation (three-phase Montgomery batch inversion)
 *
 * Optimization rationale:
 *   Original: one fe_inv (~300 multiplications) per step; 256 inversions for steps=256.
 *   Optimized using Montgomery Batch Inversion:
 *     Phase 1: Jacobian point traversal only (P += G), store each step's Z into thread-local array
 *     Phase 2: call fe_batch_inv once for all Z inverses (1 inversion + 3*(steps-1) multiplications)
 *     Phase 3: batch-compute affine coords x = X*Z_inv^2, y = Y*Z_inv^3 and write to output
 *
 * Theoretical speedup (steps=256): 256 inversions -> 1 inversion, ~98% inversion cost eliminated.
 *
 * Input:
 *   base_privkeys : [num_chains * 32] bytes, base private key for each chain (big-endian)
 * Output:
 *   aff_x, aff_y  : [num_chains * steps * 32] bytes, affine coordinates
 *   valid         : [num_chains] int, 0=invalid, >0=valid step count
 */
__global__ void kernel_gen_pubkeys(
    const uint8_t * __restrict__ base_privkeys,
    uint8_t       * __restrict__ aff_x,
    uint8_t       * __restrict__ aff_y,
    int           * __restrict__ valid,
    int num_chains,
    int steps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chains)
        return;

    const uint8_t *privkey = base_privkeys + tid * 32;

    /* Check if private key is zero (invalid) */
    int all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (privkey[i] != 0) { all_zero = 0; break; }
    }
    if (all_zero) {
        valid[tid] = 0;
        return;
    }

    /* Compute base public key: P0 = privkey * G */
    jac_point P = scalar_mult_G(privkey);
    if (P.infinity) {
        valid[tid] = 0;
        return;
    }

    /* ---- Phase 1: Jacobian point traversal, collect Jacobian coords for all steps ---- */
    /* Thread-local arrays: store Jacobian X/Y/Z for each step */
    /* GPU_CHAIN_STEPS max = 256, each fe256 = 32 bytes */
    /* 3 * 256 * 32 = 24576 bytes per thread, stored in registers/local memory */
    fe256 jac_X[GPU_CHAIN_STEPS];
    fe256 jac_Y[GPU_CHAIN_STEPS];
    fe256 jac_Z[GPU_CHAIN_STEPS];

    /* Current scalar (for overflow detection) */
    fe256 cur_scalar = fe_from_bytes(privkey);

    int valid_steps = steps;  /* default: all steps valid */

    for (int step = 0; step < steps; step++) {
        jac_X[step] = P.x;
        jac_Y[step] = P.y;
        jac_Z[step] = P.z;

        if (step < steps - 1) {
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
                valid_steps = step + 1;  /* record valid step count and truncate */
                break;
            }
        }
    }

    /* ---- Phase 2: batch modular inversion, compute all Z inverses at once ---- */
    fe256 z_inv[GPU_CHAIN_STEPS];  /* stores Z^-1 for each step */
    int batch_valid = valid_steps;  /* fe_batch_inv may truncate further if a Z is zero */
    fe_batch_inv(z_inv, jac_Z, valid_steps, &batch_valid);
    /* If batch_valid < valid_steps, some step has Z=0 (point at infinity); truncate */
    if (batch_valid < valid_steps) {
        valid_steps = batch_valid;
    }

    /* ---- Phase 3: batch-compute affine coordinates and write to output buffer ---- */
    for (int step = 0; step < valid_steps; step++) {
        fe256 zi  = z_inv[step];          /* Z^-1 */
        fe256 zi2 = fe_sqr(zi);           /* Z^-2 */
        fe256 zi3 = fe_mul(zi2, zi);      /* Z^-3 */
        fe256 ax  = fe_mul(jac_X[step], zi2);  /* x = X * Z^-2 */
        fe256 ay  = fe_mul(jac_Y[step], zi3);  /* y = Y * Z^-3 */

        int out_idx = tid * steps + step;
        fe_to_bytes(ax, aff_x + out_idx * 32);
        fe_to_bytes(ay, aff_y + out_idx * 32);
    }

    valid[tid] = valid_steps;
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

    /* Configure kernel: 256 threads per block */
    int block_size = 256;
    int grid_size  = (g_num_chains + block_size - 1) / block_size;

    kernel_gen_pubkeys<<<grid_size, block_size, 0, stream>>>(
        d_base_privkeys, d_aff_x, d_aff_y, d_valid,
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

