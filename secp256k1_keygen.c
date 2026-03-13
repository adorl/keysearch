/*
 * secp256k1_keygen.c
 *
 * Wraps libsecp256k1 internal interfaces, implementing:
 *   1. Direct point addition (bypassing ecmult path)
 *   2. Batch affine coordinate normalization (Montgomery trick)
 *   3. Direct public key byte construction from affine coordinates (skip serialize)
 *
 * Compilation modes:
 *   - Default: use internal headers, link locally compiled secp256k1_lib.o
 *   - USE_PUBKEY_API_ONLY: fall back to public API
 *
 * Note: this file defines SECP256K1_BUILD and includes *_impl.h,
 * these static/inline functions each compilation unit needs its own copy,
 * no symbol conflict with secp256k1_lib.o (static symbols are not exported).
 * Non-static symbols like secp256k1_context_struct are provided by secp256k1_lib.o.
 */

/*
 * Note: SECP256K1_BUILD and all *_impl.h are already defined/included in secp256k1_keygen.h.
 * Simply include secp256k1_keygen.h here, no need to redefine.
 */
#include "secp256k1_keygen.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef USE_PUBKEY_API_ONLY

/*
 * Full definition of secp256k1_context_struct (consistent with secp256k1/src/secp256k1.c).
 * Full definition needed here to access ctx->ecmult_gen_ctx field.
 * Struct definition produces no link symbols, no conflict with secp256k1_lib.o.
 */
struct secp256k1_context_struct {
    secp256k1_ecmult_gen_context ecmult_gen_ctx;
    secp256k1_callback illegal_callback;
    secp256k1_callback error_callback;
    int declassify;
};

int keygen_init_generator(const secp256k1_context *ctx,
                          secp256k1_ge *G_out)
{
    (void)ctx;

    /*
     * secp256k1 generator G affine coordinates (hardcoded standard values)
     * X = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
     * Y = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
     */
    static const unsigned char Gx[32] = {
        0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
        0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
        0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
        0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
    };
    static const unsigned char Gy[32] = {
        0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,
        0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
        0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,
        0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8
    };

    if (!secp256k1_fe_set_b32_limit(&G_out->x, Gx))
        return -1;
    if (!secp256k1_fe_set_b32_limit(&G_out->y, Gy))
        return -1;
    G_out->infinity = 0;

    /* Verify infinity flag */
    if (G_out->infinity != 0) {
        fprintf(stderr, "keygen_init_generator: G.infinity != 0, unexpected!\n");
        return -1;
    }
    return 0;
}

int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_gej *gej_out)
{
    secp256k1_scalar scalar;
    int overflow = 0;

    secp256k1_scalar_set_b32(&scalar, privkey, &overflow);
    if (overflow || secp256k1_scalar_is_zero(&scalar)) {
        return -1;
    }

    /* Directly call internal ecmult_gen: gej = scalar * G */
    secp256k1_ecmult_gen(&ctx->ecmult_gen_ctx, gej_out, &scalar);

    /* Clear scalar to prevent side-channel leakage */
    secp256k1_scalar_clear(&scalar);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Batch normalization using Montgomery trick:                          */
/*   acc[0] = Z[0]                                                     */
/*   acc[i] = acc[i-1] * Z[i]                                         */
/*   inv = 1 / acc[n-1]  (1 modular inverse)                          */
/*   Backward pass:                                                    */
/*     inv_zi = inv * acc[i-1]  (inverse of Z[i])                     */
/*     inv    = inv * Z[i]      (update inv to inverse of acc[i-1])   */
/* ------------------------------------------------------------------ */

/* Size consistent with BATCH_SIZE in keysearch.c */
#define KEYGEN_BATCH_MAX (4096)
static __thread secp256k1_fe acc_buf[KEYGEN_BATCH_MAX];

void keygen_batch_normalize(const secp256k1_gej *gej_in,
                            secp256k1_ge *ge_out,
                            size_t n)
{
    if (n == 0 || n > KEYGEN_BATCH_MAX)
        return;

    /* Prefer thread-local static buffer to avoid heap allocation */
    secp256k1_fe *acc = acc_buf;

    /* Step 1: compute cumulative product, skip infinity points */
    int first = 1;
    size_t first_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (gej_in[i].infinity) {
            ge_out[i].infinity = 1;
            /* acc[i] is a placeholder, not involved in computation */
            if (!first) {
                acc[i] = acc[i - 1];
            } else {
                secp256k1_fe_set_int(&acc[i], 1);
            }
            continue;
        }
        if (first) {
            acc[i] = gej_in[i].z;
            first = 0;
            first_idx = i;
        } else {
            secp256k1_fe_mul(&acc[i], &acc[i - 1], &gej_in[i].z);
        }
    }

    if (first) {
        /* All points are infinity */
        return;
    }

    /* Step 2: compute modular inverse of acc[n-1] */
    secp256k1_fe inv;
    secp256k1_fe_inv(&inv, &acc[n - 1]);

    /* Step 3: backward pass, recover inverse of each Z, convert to affine coordinates */
    for (size_t i = n; i-- > 0; ) {
        if (gej_in[i].infinity) {
            continue;
        }

        secp256k1_fe inv_zi;
        if (i == first_idx) {
            /* First non-infinity point, inv is already the inverse of Z[i] */
            inv_zi = inv;
        } else {
            /* inv_zi = inv * acc[i-1] */
            secp256k1_fe_mul(&inv_zi, &inv, &acc[i - 1]);
            /* Update inv = inv * Z[i], making it the inverse of acc[i-1] */
            secp256k1_fe_mul(&inv, &inv, &gej_in[i].z);
        }

        /* Affine coordinates: x = X/Z^2, y = Y/Z^3 */
        secp256k1_fe inv_zi2;
        secp256k1_fe_sqr(&inv_zi2, &inv_zi);

        secp256k1_fe_mul(&ge_out[i].x, &gej_in[i].x, &inv_zi2);
        secp256k1_fe_mul(&inv_zi2, &inv_zi2, &inv_zi);   /* inv_zi^3 */
        secp256k1_fe_mul(&ge_out[i].y, &gej_in[i].y, &inv_zi2);

        /* Pre-normalize x and y for downstream secp256k1_fe_get_b32 */
        secp256k1_fe_normalize_var(&ge_out[i].x);
        secp256k1_fe_normalize_var(&ge_out[i].y);
        ge_out[i].infinity = 0;
    }
}

/*
 * keygen_batch_normalize_rzr: batch normalization accelerated by rzr increment factors
 * Principle:
 *   gej_in[i] is obtained from gej_in[i-1] + G, the rzr[i-1] output by secp256k1_gej_add_ge_var
 *   satisfies Z[i] = Z[i-1] * rzr[i-1], therefore:
 *     1/Z[i-1] = (1/Z[i]) * rzr[i-1]
 *
 *   Compute modular inverse of gej_in[n-1].z once, then use rzr chain backward to derive 1/Z[i]:
 *     inv = 1/Z[n-1]
 *     inv_zi = inv  (for point n-1)
 *     inv = inv * rzr[i-1]  ->  inv = 1/Z[i-1]  (forward propagation)
 *
 * Requirement: all points are non-infinity (guaranteed by inner loop)
 * Parameters:
 *   gej_in  : input Jacobian coordinate array (size n)
 *   ge_out  : output affine coordinate array (size >= n)
 *   rzr     : Z coordinate increment factor array (size n-1, rzr[i] satisfies Z[i+1]=Z[i]*rzr[i])
 *   n       : number of array elements
 */
void keygen_batch_normalize_rzr(const secp256k1_gej *gej_in,
                                secp256k1_ge *ge_out,
                                const secp256k1_fe *rzr,
                                size_t n)
{
    if (n == 0 || n > KEYGEN_BATCH_MAX)
        return;

    /*
     * Compute modular inverse of the last point's Z coordinate (1 modular inverse)
     * No forward accumulation needed: gej_in[n-1].z is the endpoint of all rzr products
     */
    secp256k1_fe inv;
    secp256k1_fe_inv(&inv, &gej_in[n - 1].z);

    /*
     * Backward pass, use rzr chain to derive 1/Z[i] for each point, convert to affine:
     *   inv always holds 1/Z[i] for current point i
     *   After processing point i: inv = inv * rzr[i-1] = 1/Z[i-1]
     */
    for (size_t i = n; i-- > 0; ) {
        /* inv is now = 1/Z[i] */
        secp256k1_fe inv_zi2;
        secp256k1_fe_sqr(&inv_zi2, &inv);                           /* inv_zi^2 */
        secp256k1_fe_mul(&ge_out[i].x, &gej_in[i].x, &inv_zi2);     /* X/Z^2 */
        secp256k1_fe_mul(&inv_zi2, &inv_zi2, &inv);                 /* inv_zi^3 */
        secp256k1_fe_mul(&ge_out[i].y, &gej_in[i].y, &inv_zi2);     /* Y/Z^3 */

        /*
         * Pre-normalize x and y so that downstream keygen_ge_to_pubkey_bytes
         * can call secp256k1_fe_get_b32 without redundant normalize.
         * secp256k1_fe_mul output has magnitude=1 but normalized=0;
         * normalize_var is the cheapest full normalization path.
         */
        secp256k1_fe_normalize_var(&ge_out[i].x);
        secp256k1_fe_normalize_var(&ge_out[i].y);
        ge_out[i].infinity = 0;

        /* Forward propagation: inv = 1/Z[i-1] = (1/Z[i]) * rzr[i-1] */
        if (i > 0) {
            secp256k1_fe_mul(&inv, &inv, &rzr[i - 1]);
        }
    }
}

void keygen_ge_to_pubkey_bytes(const secp256k1_ge *ge,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out)
{
    if (ge->infinity)
        return;

    /*
     * Assumes ge->x and ge->y are already normalized by keygen_batch_normalize_rzr
     * or keygen_batch_normalize (via secp256k1_fe_normalize_var).
     * secp256k1_fe_get_b32 requires normalized input; in non-VERIFY builds it
     * maps directly to secp256k1_fe_impl_get_b32 (pure shift+mask, no normalize).
     *
     * x_bytes is computed once and shared by both compressed and uncompressed paths.
     */
    uint8_t x_bytes[32];
    secp256k1_fe_get_b32(x_bytes, &ge->x);

    if (compressed_out != NULL) {
        /* Compressed pubkey: prefix 0x02 (Y even) or 0x03 (Y odd) + X (32 bytes) */
        compressed_out[0] = secp256k1_fe_is_odd(&ge->y) ? 0x03 : 0x02;
        memcpy(compressed_out + 1, x_bytes, 32);
    }

    if (uncompressed_out != NULL) {
        uint8_t y_bytes[32];
        secp256k1_fe_get_b32(y_bytes, &ge->y);
        /* Uncompressed pubkey: 0x04 + X (32 bytes) + Y (32 bytes) */
        uncompressed_out[0] = 0x04;
        memcpy(uncompressed_out + 1, x_bytes, 32);
        memcpy(uncompressed_out + 1 + 32, y_bytes, 32);
    }
}

#else

int keygen_init_generator(const secp256k1_context *ctx,
                          void *G_out)
{
    (void)ctx;
    (void)G_out;
    /* Fallback mode: no need to preload G, return success directly */
    return 0;
}

int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_pubkey *pubkey_out)
{
    if (!secp256k1_ec_pubkey_create(ctx, pubkey_out, privkey))
        return -1;
    return 0;
}

void keygen_ge_to_pubkey_bytes(const secp256k1_context *ctx,
                               const secp256k1_pubkey *pubkey,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out)
{
    if (compressed_out != NULL) {
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(ctx, compressed_out, &len,
                                      pubkey, SECP256K1_EC_COMPRESSED);
    }
    if (uncompressed_out != NULL) {
        size_t len = 65;
        secp256k1_ec_pubkey_serialize(ctx, uncompressed_out, &len,
                                      pubkey, SECP256K1_EC_UNCOMPRESSED);
    }
}

#endif /* USE_PUBKEY_API_ONLY */

