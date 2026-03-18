#ifndef SECP256K1_KEYGEN_H
#define SECP256K1_KEYGEN_H

#include <stdint.h>
#include <stddef.h>

/*
 * secp256k1_keygen.h
 *
 * Provides:
 *   1. Direct point addition
 *   2. Batch affine coordinate normalization (Batch Normalization)
 *   3. Direct public key byte construction from affine coordinates (skip serialize)
 *
 * Compilation modes:
 *   - Default: include libsecp256k1 internal headers
 *   - USE_PUBKEY_API_ONLY: fall back to public API
 */

#ifndef USE_PUBKEY_API_ONLY

/*
 * Internal interface mode: directly use libsecp256k1 internal types and implementations.
 * Compile with -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC)
 * Link with locally compiled secp256k1_lib.o
 * and precomputed_ecmult.o and precomputed_ecmult_gen.o
 */

/* Must define SECP256K1_BUILD before including any secp256k1 headers */
#ifndef SECP256K1_BUILD
#  define SECP256K1_BUILD
#endif

#include "../include/secp256k1.h"

/* Include internal type declarations and static function implementations */
#include "assumptions.h"
#include "util.h"
#include "field.h"
#include "field_impl.h"
#include "scalar.h"
#include "scalar_impl.h"
#include "group.h"
#include "group_impl.h"
#include "ecmult_gen.h"
#include "ecmult_gen_impl.h"
#include "int128_impl.h"
/*
 * Initialize global generator G affine coordinates
 * Must be called once before all threads start
 * Return value: 0 success, -1 failure (infinity flag abnormal)
 */
int keygen_init_generator(const secp256k1_context *ctx,
                          secp256k1_ge *G_out);

/*
 * Generate Jacobian coordinate public key (secp256k1_gej) from 32-byte private key
 * Internally calls secp256k1_ecmult_gen, called only once at the start of each batch
 * Return value: 0 success, -1 failure
 */
int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_gej *gej_out);

/*
 * Batch normalize Jacobian coordinate array to affine coordinate array
 * Uses Montgomery trick: only 1 modular inverse + 3*(n-1) multiplications.
 * Infinity points are skipped (corresponding ge_out[i].infinity == 1)
 * Parameters:
 *   gej_in  : input Jacobian coordinate array
 *   ge_out  : output affine coordinate array (caller-allocated, size >= n)
 *   n       : number of array elements
 */
void keygen_batch_normalize(const secp256k1_gej *gej_in,
                            secp256k1_ge *ge_out,
                            size_t n);

/*
 * Batch normalization accelerated by rzr increment factors (avoids memory reads of gej.z in forward accumulation)
 * rzr[i] satisfies: Z[i+1] = Z[i] * rzr[i] (provided by rzr parameter of secp256k1_gej_add_ge_var)
 * Requirement: all points are non-infinity (guaranteed by inner loop), rzr array size is n-1
 * Parameters:
 *   gej_in  : input Jacobian coordinate array (size n)
 *   ge_out  : output affine coordinate array (caller-allocated, size >= n)
 *   rzr     : Z coordinate increment factor array (size n-1, rzr[i] corresponds to Z increment from gej_in[i] to gej_in[i+1])
 *   n       : number of array elements
 */
void keygen_batch_normalize_rzr(const secp256k1_gej *gej_in,
                                secp256k1_ge *ge_out,
                                const secp256k1_fe *rzr,
                                size_t n);

/*
 * Directly construct compressed/uncompressed public key bytes from affine coordinates, skip serialize call
 * Must ensure ge->infinity == 0 before calling
 * Parameters:
 *   ge              : normalized affine coordinate point
 *   compressed_out  : compressed pubkey output (33 bytes), pass NULL to skip
 *   uncompressed_out: uncompressed pubkey output (65 bytes), pass NULL to skip
 */
void keygen_ge_to_pubkey_bytes(const secp256k1_ge *ge,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out);

#ifdef __AVX512IFMA__

#include <immintrin.h>

/* Forward declaration of SoA types used in AVX-512 IFMA path */
typedef struct {
    __m512i n[5][2];
} secp256k1_fe_16x;

typedef struct {
    secp256k1_fe_16x x;
    secp256k1_fe_16x y;
    secp256k1_fe_16x z;
} secp256k1_gej_16x;

/* AVX-512 IFMA 16-way parallel point addition (AoS interface)
 * normed=0: apply normalize_weak to input coordinates (used for first step)
 * normed=1: skip normalize_weak (caller guarantees magnitude=1, used for subsequent steps)
 */
void gej_add_ge_var_16way(secp256k1_gej r[16],
                          const secp256k1_gej a[16],
                          const secp256k1_ge *b,
                          secp256k1_fe rzr[16],
                          int normed);

/* AVX-512 IFMA 16-way parallel point addition (SoA persistent interface)
 * Operates directly on SoA layout, eliminating AoS<->SoA conversion in hot loops.
 * r_aos may be NULL if AoS output is not needed for that call.
 */
void gej_add_ge_var_16way_soa(secp256k1_gej_16x *r_soa,
                              const secp256k1_gej_16x *a_soa,
                              const secp256k1_ge *b,
                              secp256k1_gej *r_aos_ptrs[16],
                              secp256k1_fe rzr_out[16],
                              int normed);

/* Convert 16 secp256k1_gej elements (AoS) to SoA layout */
void gej_16x_load(secp256k1_gej_16x *dst, const secp256k1_gej src[16]);

/* Convert SoA layout back to 16 secp256k1_gej elements (AoS) */
void gej_16x_store(secp256k1_gej dst[16], const secp256k1_gej_16x *src);

/* Scatter-write SoA layout to 16 non-contiguous secp256k1_gej pointers */
void gej_16x_store_scatter(secp256k1_gej *dst_ptrs[16], const secp256k1_gej_16x *src);

/* Batch convert 16 normalized secp256k1_fe to 32-byte big-endian using AVX-512 */
void fe_get_b32_16way(uint8_t out[16][32], const secp256k1_fe fe_arr[16]);

/* Convert 16 scattered secp256k1_fe pointers to SoA layout (no intermediate copy) */
void fe_16x_load_ptrs(secp256k1_fe_16x *dst, const secp256k1_fe *src_ptrs[16]);

/* Direct SoA fe → SHA-256 big-endian SoA words conversion (16-way, pure SIMD) */
void fe_to_sha256_words_16way(const secp256k1_fe_16x *fe, __m512i w[8]);

/* Batch convert 16 affine points to compressed/uncompressed pubkey bytes using AVX-512 */
void keygen_ge_to_pubkey_bytes_16way(const secp256k1_ge ge[16],
                                     uint8_t *compressed_out[16],
                                     uint8_t *uncompressed_out[16]);

/*
 * 16-chain batch normalization with direct SoA output (eliminates fe_16x_load_ptrs).
 *
 * Processes 16 independent chains of Jacobian points using rzr acceleration,
 * and outputs affine coordinates directly in SoA layout (secp256k1_fe_16x per step).
 *
 * Parameters:
 *   gej_in       : flat array of 16*n Jacobian points, row-major [ch*n + step]
 *   fe_x_soa_out : output X coordinate array in SoA layout (size n)
 *   fe_y_soa_out : output Y coordinate array in SoA layout (size n)
 *   rzr          : flat array of 16*n Z increment factors, row-major [ch*n + step]
 *   ge_work      : caller-allocated work buffer for 16*n affine points (avoids malloc per call)
 *   valid_steps  : [16] valid step count per chain
 *   n            : max steps (must be >= max(valid_steps[ch]))
 */
void keygen_batch_normalize_rzr_16way(const secp256k1_gej *gej_in,
    secp256k1_fe_16x *fe_x_soa_out, secp256k1_fe_16x *fe_y_soa_out, const secp256k1_fe *rzr,
    secp256k1_ge *ge_work, const int *valid_steps, size_t n);

#endif /* __AVX512IFMA__ */

#else

/* Fallback mode: use system-installed secp256k1.h for public types */
#include <secp256k1.h>

/*
 * Fallback mode: initialization (no-op, for compatibility only)
 * Return value: 0 success
 */
int keygen_init_generator(const secp256k1_context *ctx,
                          void *G_out);

/*
 * Fallback mode: create public key from private key (using secp256k1_ec_pubkey_create)
 * gej_out is actually secp256k1_pubkey*
 * Return value: 0 success, -1 failure
 */
int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_pubkey *pubkey_out);

/*
 * Fallback mode: serialize single public key to compressed/uncompressed bytes
 * Parameters:
 *   pubkey          : secp256k1_pubkey pointer
 *   compressed_out  : compressed pubkey output (33 bytes), pass NULL to skip
 *   uncompressed_out: uncompressed pubkey output (65 bytes), pass NULL to skip
 */
void keygen_ge_to_pubkey_bytes(const secp256k1_context *ctx,
                               const secp256k1_pubkey *pubkey,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out);

#endif /* USE_PUBKEY_API_ONLY */

#endif /* SECP256K1_KEYGEN_H */


