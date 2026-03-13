/*
 * Dependencies:
 *   - libsecp256k1  (elliptic curve public key computation)
 *   - pthread       (multi-threading)
 *   SHA256 and RIPEMD160 are embedded pure-C implementations, no OpenSSL required
 * Usage:
 *   ./keysearch -a <address_file> [-n num_threads] [-h]
 */
#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include "keylog.h"
#include "sha256.h"
#include "ripemd160.h"
#include "hash_utils.h"
#include "rand_key.h"

#ifdef USE_GPU
#  include "gpu/gpu_interface.h"
#endif
/* secp256k1_keygen.h already includes secp256k1.h in internal mode
 * Fallback mode requires system secp256k1.h, handled via conditional compilation */
#ifndef USE_PUBKEY_API_ONLY
#  include "secp256k1_keygen.h"
#else
#  include <secp256k1.h>
#  include "secp256k1_keygen.h"
#endif


#define MAX_ATTEMPTS        (1ULL << 63)    /* max attempts per thread */
#define PROGRESS_INTERVAL   (10000000)      /* progress print interval */
#define MAX_ADDRESSES       (400000)        /* max number of target addresses */
#define ADDRESS_LEN         (35)            /* max Bitcoin address length */
#define BATCH_SIZE          (65536)         /* incremental derivation batch size, reset random base key after each batch */


static int address_count = 0;                /* number of loaded addresses */

/* cross-thread found flag */
static volatile int found_flag = 0;

#ifdef USE_GPU
/* GPU path flag (set by -g argument) */
static int use_gpu = 0;
#endif

/* secp256k1 context (read-only, thread-safe) */
secp256k1_context *secp_ctx = NULL;

#ifndef USE_PUBKEY_API_ONLY
/* Global generator G affine coordinates (initialized by keygen_init_generator) */
secp256k1_ge G_affine;
#endif
struct thread_args
{
    int thread_id;
};


#ifndef USE_PUBKEY_API_ONLY
/*
 * Fast scalar increment by 1: directly add 1 to d[0] and propagate carry.
 * Avoids the overhead of full 128-bit addition in secp256k1_scalar_add.
 * Returns 1 if the result reduces to zero (i.e., input was N-1).
 *
 * The secp256k1 order N = {0xBFD25E8CD0364141, 0xBAAEDCE6AF48A03B,
 *   0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF}.
 * N_0 = 0xBFD25E8CD0364141, so d[0] wraps only when old d[0] = 0xFFFFFFFFFFFFFFFF.
 * Most increments (>99.99%) hit the fast path: ++d[0], check_overflow, done.
 *
 * Note: we MUST check overflow even on the fast path, because s could be N-1
 * where ++d[0] does not wrap but the result equals N.
 */
static inline int scalar_increment(secp256k1_scalar *s) {
    int overflow;
    if (__builtin_expect(++s->d[0] != 0, 1)) {
        /* Fast path: no carry propagation needed, just check overflow */
        overflow = secp256k1_scalar_check_overflow(s);
        if (__builtin_expect(overflow, 0)) {
            secp256k1_scalar_reduce(s, overflow);
            return secp256k1_scalar_is_zero(s);
        }
        return 0;
    }
    if (__builtin_expect(++s->d[1] != 0, 1))
        goto check;
    if (__builtin_expect(++s->d[2] != 0, 1))
        goto check;
    ++s->d[3];
check:
    overflow = secp256k1_scalar_check_overflow(s);
    if (overflow) {
        secp256k1_scalar_reduce(s, overflow);
        return secp256k1_scalar_is_zero(s);
    }
    return 0;
}
#endif /* USE_PUBKEY_API_ONLY */

static void *search_key(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    int thread_id = args->thread_id;

    uint8_t privkey[32];            /* current private key (random base per batch, incremented in inner loop) */
    uint8_t hash160_compressed[20];
    uint8_t hash160_uncompressed[20];
#ifndef USE_PUBKEY_API_ONLY
    /* AVX2/scalar internal path: pre-construct tweak scalar (value=1), initialized once before outer loop */    secp256k1_scalar tweak_scalar;
    secp256k1_scalar_set_int(&tweak_scalar, 1);
#else
    uint8_t tweak[32];              /* scalar addition tweak = 1 (used in fallback path) */
    memset(tweak, 0, 32);
    tweak[31] = 1;
#endif
    uint64_t count = 0;
    int progress_counter = PROGRESS_INTERVAL; /* countdown counter, avoids modulo division */
    rand_key_context rand_ctx;

    /* Initialize true-random context (per-thread independent fd, no lock contention) */
    if (rand_ctx_init(&rand_ctx) != 0) {
        keylog_error("[Thread-%d] Failed to open /dev/urandom", thread_id);
        return NULL;
    }

    /* Performance monitoring: record last print time */
    struct timespec ts_last, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_last);
    uint64_t count_last = 0;

#ifndef USE_PUBKEY_API_ONLY

#ifdef __AVX512IFMA__
    /*
     * AVX-512 IFMA 16-chain buffers: each chain accumulates BATCH_SIZE steps of Jacobian points and rzr factors
     * Three buffers total ~24MB, must be heap-allocated
     * Allocated once before the outermost loop, freed when thread exits
     */    secp256k1_gej (*gej_buf)[BATCH_SIZE] = malloc(16 * BATCH_SIZE * sizeof(secp256k1_gej));
    secp256k1_fe (*rzr_buf)[BATCH_SIZE] = malloc(16 * BATCH_SIZE * sizeof(secp256k1_fe));
    secp256k1_ge (*ge_buf)[BATCH_SIZE] = malloc(16 * BATCH_SIZE * sizeof(secp256k1_ge));
    if (!gej_buf || !rzr_buf || !ge_buf) {
        keylog_error("[Thread-%d] AVX-512 batch buffer memory allocation failed", thread_id);
        goto avx512_thread_exit;
    }

    while (count < MAX_ATTEMPTS) {
        /* Current Jacobian coordinates and private key scalar for 16 independent chains */
        secp256k1_gej chain_gej[16];
        secp256k1_scalar chain_scalar[16];
        secp256k1_scalar chain_base_scalar[16];
        /* chain_valid_steps[ch]: actual valid steps accumulated in this batch for this chain */
        int chain_valid_steps[16];

        /* Initialize 16 chains: all independently randomly generated */
        for (int ch = 0; ch < 16; ch++) {
            uint8_t ch_privkey[32];
            int ch_ok = 0;
            while (!ch_ok) {
                if (gen_random_key(ch_privkey, &rand_ctx) != 0) {
                    keylog_error("[Thread-%d] Failed to read random number", thread_id);                    continue;
                }

                int overflow = 0;
                secp256k1_scalar_set_b32(&chain_base_scalar[ch], ch_privkey, &overflow);
                if (overflow || secp256k1_scalar_is_zero(&chain_base_scalar[ch]))
                    continue;
                if (keygen_privkey_to_gej(secp_ctx, ch_privkey, &chain_gej[ch]) != 0)
                    continue;

                chain_scalar[ch] = chain_base_scalar[ch];
                ch_ok = 1;
            }

            chain_valid_steps[ch] = BATCH_SIZE;
        }

        /* Inner loop: advance 16 chains one step each, BATCH_SIZE steps total, accumulate without normalizing */
        for (int step = 0; step < BATCH_SIZE; step++) {
            secp256k1_fe step_rzr[16];
            secp256k1_gej next_gej_16[16];
            if (step == 0)
                gej_add_ge_var_16way(next_gej_16, chain_gej, &G_affine, step_rzr, 0);
            else
                gej_add_ge_var_16way(next_gej_16, chain_gej, &G_affine, step_rzr, 1);

            for (int ch = 0; ch < 16; ch++) {
                /* chain has already overflowed to zero */
                if (step >= chain_valid_steps[ch]) {
                    continue;
                }

                gej_buf[ch][step] = next_gej_16[ch];
                rzr_buf[ch][step] = step_rzr[ch];

                /* Update each chain state (scalar+1, detect overflow to zero) */
                if (__builtin_expect(scalar_increment(&chain_scalar[ch]), 0)) {
                    chain_valid_steps[ch] = step + 1;
                } else {
                    chain_gej[ch] = next_gej_16[ch];
                }
            }
        }

        /* Batch normalize: call keygen_batch_normalize_rzr for each chain */
        for (int ch = 0; ch < 16; ch++) {
            keygen_batch_normalize_rzr(gej_buf[ch], ge_buf[ch], rzr_buf[ch],
                                       (size_t)chain_valid_steps[ch]);
        }

        /* Iterate all steps, compute hash160 and lookup in batches of 16 */
        for (int step = 0; step < BATCH_SIZE && count < MAX_ATTEMPTS; step++) {
            uint8_t comp_bufs[16][64];
            uint8_t uncomp_bufs[16][128];
            const uint8_t *comp_ptrs[16];
            const uint8_t *uncomp_ptrs[16];

            for (int lane = 0; lane < 16; lane++) {
                if (step >= chain_valid_steps[lane] || ge_buf[lane][step].infinity) {
                    memset(comp_bufs[lane], 0, 64);
                    memset(uncomp_bufs[lane], 0, 128);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                } else {
                    keygen_ge_to_pubkey_bytes(&ge_buf[lane][step],
                                             comp_bufs[lane], uncomp_bufs[lane]);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                comp_ptrs[lane] = comp_bufs[lane];
                uncomp_ptrs[lane] = uncomp_bufs[lane];
            }

            /* 16-way parallel hash160 computation */
            uint8_t hash160_comp_16[16][20];
            uint8_t hash160_uncomp_16[16][20];
            hash160_16way_compressed_prepadded(comp_ptrs, hash160_comp_16);
            hash160_16way_uncompressed_prepadded(uncomp_ptrs, hash160_uncomp_16);

            const uint8_t *comp_h160_ptrs[16];
            const uint8_t *uncomp_h160_ptrs[16];
            for (int lane = 0; lane < 16; lane++) {
                comp_h160_ptrs[lane] = hash160_comp_16[lane];
                uncomp_h160_ptrs[lane] = hash160_uncomp_16[lane];
            }

            /* 16-way batch lookup */
            uint16_t mask_comp = ht_contains_16way(comp_h160_ptrs);
            uint16_t mask_uncomp = ht_contains_16way(uncomp_h160_ptrs);
            uint16_t hit_mask = mask_comp | mask_uncomp;

            /* Count valid lanes in this step */
            int valid_lanes = 0;
            for (int lane = 0; lane < 16; lane++) {
                if (step < chain_valid_steps[lane])
                    valid_lanes++;
            }
            count += (uint64_t)valid_lanes;

            if (hit_mask) {
                for (int lane = 0; lane < 16; lane++) {
                    if (!(hit_mask & (1u << lane)))
                        continue;
                    if (step >= chain_valid_steps[lane])
                        continue;
                    if (ge_buf[lane][step].infinity)
                        continue;

                    found_flag = 1;
                    secp256k1_scalar hit_scalar = chain_base_scalar[lane];
                    for (int i = 0; i < step; i++) {
                        secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
                    }
                    secp256k1_scalar_get_b32(privkey, &hit_scalar);
                    char privkey_hex[65];
                    char address_compressed[ADDRESS_LEN + 1];
                    char address_uncompressed[ADDRESS_LEN + 1];
                    bytes_to_hex(privkey, 32, privkey_hex);
                    keylog_info("[Thread-%d] Found match! Total attempts: %lu", thread_id, count);
                    keylog_info("Private key (hex): %s", privkey_hex);
                    if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                        keylog_info("Compressed address: %s", address_compressed);
                        keylog_info("Uncompressed address: %s", address_uncompressed);
                    }
                }
            }

            /* Performance monitoring */
            progress_counter -= valid_lanes;
            if (progress_counter <= 0) {
                progress_counter = PROGRESS_INTERVAL;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec) + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                keylog_info("[Thread-%d] Attempted: %lu Speed: %.0f keys/s", thread_id, count, kps);
                ts_last = ts_now;
                count_last = count;
            }
        }
    }

avx512_thread_exit:
    free(gej_buf);
    free(rzr_buf);
    free(ge_buf);

#else /* !__AVX512IFMA__ */

    secp256k1_gej *gej_batch = malloc(BATCH_SIZE * sizeof(secp256k1_gej));  /* Jacobian coordinate batch buffer */
    secp256k1_ge *ge_batch = malloc(BATCH_SIZE * sizeof(secp256k1_ge));     /* affine coordinate batch buffer */
    secp256k1_fe *rzr_batch = malloc(BATCH_SIZE * sizeof(secp256k1_fe));    /* Z coordinate increment factor: Z[i+1] = Z[i] * rzr[i] */
    if (!gej_batch || !ge_batch || !rzr_batch) {
        keylog_error("[Thread-%d] Batch buffer memory allocation failed", thread_id);
        goto thread_exit;
    }

    uint8_t pubkey_compressed[33];
    uint8_t pubkey_uncompressed[65];

    while (count < MAX_ATTEMPTS) {
        /* Generate random base private key */
        if (gen_random_key(privkey, &rand_ctx) != 0) {
            keylog_error("[Thread-%d] Failed to read random number", thread_id);
            break;
        }

        /* Convert base private key to scalar form, inner loop accumulates directly on scalar */
        secp256k1_scalar base_privkey_scalar;
        secp256k1_scalar cur_privkey_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_privkey_scalar, privkey, &overflow);
        if (overflow || secp256k1_scalar_is_zero(&base_privkey_scalar))
            continue;

        cur_privkey_scalar = base_privkey_scalar;

        secp256k1_gej cur_gej;
        secp256k1_gej next_gej;

        /* Generate Jacobian coordinate public key from base private key */
        if (keygen_privkey_to_gej(secp_ctx, privkey, &cur_gej) != 0)
            continue;

        /* Inner loop: accumulate BATCH_SIZE Jacobian points */
        int batch_valid = 0;
        int inner_overflow = 0; /* flag whether inner scalar addition overflowed */
        for (int b = 0; b < BATCH_SIZE && count < MAX_ATTEMPTS; b++) {
            gej_batch[b] = cur_gej;
            batch_valid++;

            /* last point does not need to derive next step */
            if (b == BATCH_SIZE - 1)
                break;

            /* Incremental derivation: private key scalar+1, public key point add G (direct point addition, no ecmult) */
            if (__builtin_expect(scalar_increment(&cur_privkey_scalar), 0)) {
                /* Very rare: scalar overflows to zero, break inner loop and regenerate base private key */
                inner_overflow = 1;
                break;
            }
            /* Use variable-time point addition, collect Z coordinate increment factor rzr[b] for batch_normalize acceleration */
            secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
            cur_gej = next_gej;
        }

        if (inner_overflow)
            continue;

        /* Batch normalize: use rzr increment factors, avoid memory reads of gej.z in forward accumulation */
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)batch_valid);

#ifdef __AVX512F__

        /* Process in batches of 16 (AVX-512F 16-way concurrent SHA256+RIPEMD160) */
        for (int b = 0; b < batch_valid; b += 16) {
            /* Compute actual valid lane count in this group (pad with last valid point if < 16) */
            int valid_count = batch_valid - b;
            if (valid_count > 16)
                valid_count = 16;

            uint8_t comp_bufs[16][64];
            uint8_t uncomp_bufs[16][128];
            const uint8_t *comp_ptrs[16];
            const uint8_t *uncomp_ptrs[16];

            for (int lane = 0; lane < 16; lane++) {
                int idx = b + (lane < valid_count ? lane : valid_count - 1);
                if (ge_batch[idx].infinity) {
                    memset(comp_bufs[lane], 0, 64);
                    memset(uncomp_bufs[lane], 0, 128);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                else {
                    keygen_ge_to_pubkey_bytes(&ge_batch[idx], comp_bufs[lane], uncomp_bufs[lane]);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                comp_ptrs[lane] = comp_bufs[lane];
                uncomp_ptrs[lane] = uncomp_bufs[lane];
            }

            /* 16-way parallel hash160 computation (both compressed/uncompressed use pre-padded interface, zero-copy) */
            uint8_t hash160_comp_16[16][20];
            uint8_t hash160_uncomp_16[16][20];
            hash160_16way_compressed_prepadded(comp_ptrs, hash160_comp_16);
            hash160_16way_uncompressed_prepadded(uncomp_ptrs, hash160_uncomp_16);
            const uint8_t *comp_h160_ptrs[16];
            const uint8_t *uncomp_h160_ptrs[16];
            for (int lane = 0; lane < 16; lane++) {
                comp_h160_ptrs[lane] = hash160_comp_16[lane];
                uncomp_h160_ptrs[lane] = hash160_uncomp_16[lane];
            }

            /* 16-way batch lookup (compressed + uncompressed each once) */
            uint16_t mask_comp = ht_contains_16way(comp_h160_ptrs);
            uint16_t mask_uncomp = ht_contains_16way(uncomp_h160_ptrs);
            uint16_t hit_mask = mask_comp | mask_uncomp;

            /* Update count (valid lanes only) */
            count += (uint64_t)valid_count;

            /* Only enter processing logic when there is a hit */
            if (hit_mask) {
                for (int lane = 0; lane < valid_count; lane++) {
                    if (!(hit_mask & (1u << lane)))
                        continue;

                    int b_idx = b + lane;
                    if (ge_batch[b_idx].infinity)
                        continue;

                    found_flag = 1;
                    secp256k1_scalar hit_scalar = base_privkey_scalar;
                    for (int i = 0; i < b_idx; i++) {
                        secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
                    }
                    secp256k1_scalar_get_b32(privkey, &hit_scalar);
                    char privkey_hex[65];
                    char address_compressed[ADDRESS_LEN + 1];
                    char address_uncompressed[ADDRESS_LEN + 1];
                    bytes_to_hex(privkey, 32, privkey_hex);
                    keylog_info("[Thread-%d] Found match! Total attempts: %lu", thread_id, count);
                    keylog_info("Private key (hex): %s", privkey_hex);
                    if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                        keylog_info("Compressed address: %s", address_compressed);
                        keylog_info("Uncompressed address: %s", address_uncompressed);
                    }
                }
            }

            progress_counter -= valid_count;
            if (progress_counter <= 0) {
                progress_counter = PROGRESS_INTERVAL;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec) + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                keylog_info("[Thread-%d] Attempted: %lu Speed: %.0f keys/s", thread_id, count, kps);
                ts_last = ts_now;
                count_last = count;
            }
        }
#elif defined(__AVX2__)

        /* Process in batches of 8 */
        for (int b = 0; b < batch_valid; b += 8) {
            /* Compute actual valid lane count in this group (pad with last valid point if < 8) */
            int valid_count = batch_valid - b;
            if (valid_count > 8)
                valid_count = 8;

            /* comp_bufs: 64 bytes (first 33 bytes = pubkey, last 31 bytes = SHA256 padding)
             * uncomp_bufs: 128 bytes (first 65 bytes = pubkey, last 63 bytes = SHA256 block2 padding)
             * Both construct padded block in-place, avoiding internal copy in hash160 function */
            uint8_t comp_bufs[8][64];
            uint8_t uncomp_bufs[8][128];
            const uint8_t *comp_ptrs[8];
            const uint8_t *uncomp_ptrs[8];

            for (int lane = 0; lane < 8; lane++) {
                int idx = b + (lane < valid_count ? lane : valid_count - 1);
                if (ge_batch[idx].infinity) {
                    /* Infinity point: fill with zero pubkey and pad (won't hit hash table) */
                    memset(comp_bufs[lane], 0, 64);
                    memset(uncomp_bufs[lane], 0, 128);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                else {
                    keygen_ge_to_pubkey_bytes(&ge_batch[idx], comp_bufs[lane], uncomp_bufs[lane]);
                    /* Complete SHA256 padding in-place, avoiding memset+memcpy inside hash160 function */
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                comp_ptrs[lane] = comp_bufs[lane];
                uncomp_ptrs[lane] = uncomp_bufs[lane];
            }

            /* 8-way parallel hash160 computation (both compressed/uncompressed use pre-padded interface, zero-copy) */
            uint8_t hash160_comp_8[8][20];
            uint8_t hash160_uncomp_8[8][20];
            hash160_8way_compressed_prepadded(comp_ptrs, hash160_comp_8);
            hash160_8way_uncompressed_prepadded(uncomp_ptrs, hash160_uncomp_8);
            const uint8_t *comp_h160_ptrs[8];
            const uint8_t *uncomp_h160_ptrs[8];
            for (int lane = 0; lane < 8; lane++) {
                comp_h160_ptrs[lane] = hash160_comp_8[lane];
                uncomp_h160_ptrs[lane] = hash160_uncomp_8[lane];
            }

            /* 8-way batch lookup (compressed + uncompressed each once, 16 total) */
            uint8_t mask_comp = ht_contains_8way(comp_h160_ptrs);
            uint8_t mask_uncomp = ht_contains_8way(uncomp_h160_ptrs);
            uint8_t hit_mask = mask_comp | mask_uncomp;

            /* Update count (valid lanes only) */
            count += (uint64_t)valid_count;

            /* Only enter processing logic when there is a hit */
            if (hit_mask) {
                for (int lane = 0; lane < valid_count; lane++) {
                    if (!(hit_mask & (1 << lane)))
                        continue;

                    int b_idx = b + lane;
                    if (ge_batch[b_idx].infinity)
                        continue;

                    found_flag = 1;
                    /* Rebuild private key on hit: start from base_privkey_scalar, use scalar addition to rebuild hit position */
                    secp256k1_scalar hit_scalar = base_privkey_scalar;
                    for (int i = 0; i < b_idx; i++) {
                        secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
                    }
                    secp256k1_scalar_get_b32(privkey, &hit_scalar);
                    char privkey_hex[65];
                    char address_compressed[ADDRESS_LEN + 1];
                    char address_uncompressed[ADDRESS_LEN + 1];
                    bytes_to_hex(privkey, 32, privkey_hex);
                    keylog_info("[Thread-%d] Found match! Total attempts: %lu", thread_id, count);
                    keylog_info("Private key (hex): %s", privkey_hex);
                    if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                        keylog_info("Compressed address: %s", address_compressed);
                        keylog_info("Uncompressed address: %s", address_uncompressed);
                    }
                }
            }

            /* Performance monitoring (decremented by batch) */
            progress_counter -= valid_count;
            if (progress_counter <= 0) {
                progress_counter = PROGRESS_INTERVAL;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec) + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                keylog_info("[Thread-%d] Attempted: %lu Speed: %.0f keys/s", thread_id, count, kps);
                ts_last = ts_now;
                count_last = count;
            }
        }
#else
        /* Scalar path (non-AVX512F/AVX2 platform) */
        for (int b = 0; b < batch_valid; b++) {
            count++;

            if (ge_batch[b].infinity)
                continue;

            /* Construct public key bytes directly from affine coordinates, skip serialize */
            keygen_ge_to_pubkey_bytes(&ge_batch[b],
                                      pubkey_compressed,
                                      pubkey_uncompressed);

            /* Compute hash160 */
            pubkey_bytes_to_hash160(pubkey_compressed, 33, hash160_compressed);
            pubkey_bytes_to_hash160(pubkey_uncompressed, 65, hash160_uncompressed);

            /* Hash table lookup (direct byte comparison) */
            if (ht_contains(hash160_compressed) || ht_contains(hash160_uncompressed)) {
                found_flag = 1;
                /* Rebuild private key on hit: start from base_privkey_scalar, use scalar addition to rebuild hit position */
                secp256k1_scalar hit_scalar = base_privkey_scalar;
                for (int i = 0; i < b; i++) {
                    secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
                }
                secp256k1_scalar_get_b32(privkey, &hit_scalar);
                /* Only do format conversion on hit */
                char privkey_hex[65];
                char address_compressed[ADDRESS_LEN + 1];
                char address_uncompressed[ADDRESS_LEN + 1];
                bytes_to_hex(privkey, 32, privkey_hex);
                keylog_info("[Thread-%d] Found match! Total attempts: %lu", thread_id, count);
                keylog_info("Private key (hex): %s", privkey_hex);
                if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                    keylog_info("Compressed address: %s", address_compressed);
                    keylog_info("Uncompressed address: %s", address_uncompressed);
                }
            }

            /* Performance monitoring: output keys/s every PROGRESS_INTERVAL iterations */
            if (--progress_counter == 0) {
                progress_counter = PROGRESS_INTERVAL;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec) + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                keylog_info("[Thread-%d] Attempted: %lu Speed: %.0f keys/s", thread_id, count, kps);
                ts_last = ts_now;
                count_last = count;
            }
        }
#endif /* __AVX512F__ / __AVX2__ */
    }

thread_exit:
    free(gej_batch);
    free(ge_batch);
    free(rzr_batch);

#endif /* __AVX512IFMA__ */

#else /* USE_PUBKEY_API_ONLY */

    secp256k1_pubkey pubkey;
    uint8_t pubkey_compressed[33];
    uint8_t pubkey_uncompressed[65];

    while (count < MAX_ATTEMPTS) {
        /* Generate random private key */
        if (gen_random_key(privkey, &rand_ctx) != 0) {
            keylog_error("[Thread-%d] Failed to read random number", thread_id);
            break;
        }

        if (!secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey))
            continue;

        /* Inner loop: incremental derivation BATCH_SIZE times */
        for (int batch = 0; batch < BATCH_SIZE && count < MAX_ATTEMPTS; batch++) {
            count++;

            size_t len_comp = 33, len_uncomp = 65;
            secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_compressed, &len_comp,
                                          &pubkey, SECP256K1_EC_COMPRESSED);
            secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_uncompressed, &len_uncomp,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);

            pubkey_bytes_to_hash160(pubkey_compressed,   33, hash160_compressed);
            pubkey_bytes_to_hash160(pubkey_uncompressed, 65, hash160_uncompressed);

            if (ht_contains(hash160_compressed) || ht_contains(hash160_uncompressed)) {
                found_flag = 1;
                char privkey_hex[65];
                char address_compressed[ADDRESS_LEN + 1];
                char address_uncompressed[ADDRESS_LEN + 1];
                bytes_to_hex(privkey, 32, privkey_hex);
                keylog_info("[Thread-%d] Found match! Total attempts: %lu", thread_id, count);
                keylog_info("Private key (hex): %s", privkey_hex);
                if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                    keylog_info("Compressed address: %s", address_compressed);
                    keylog_info("Uncompressed address: %s", address_uncompressed);
                }
            }

            if (--progress_counter == 0) {
                progress_counter = PROGRESS_INTERVAL;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec)
                               + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                keylog_info("[Thread-%d] Attempted: %lu Speed: %.0f keys/s",
                        thread_id, count, kps);
                ts_last = ts_now;
                count_last = count;
            }

            if (!secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak)) {
                keylog_warn("Private key derivation failed, batch=%d!", batch);
                break;
            }
            if (!secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak)) {
                keylog_warn("Public key derivation failed, batch=%d!", batch);
                break;
            }
        }
    }
#endif /* USE_PUBKEY_API_ONLY */

    if (count >= MAX_ATTEMPTS) {
        keylog_info("[Thread-%d] Reached max attempts, exiting.", thread_id);
    }

    return NULL;
}


static int load_target_addresses(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        keylog_error("File %s does not exist!", filename);
        return -1;
    }

    char line[ADDRESS_LEN + 2];
    int count = 0;
    int skip_count = 0;
    while (fgets(line, sizeof(line), f)) {
        /* Strip newline characters */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0)
            continue;
        if (count >= MAX_ADDRESSES) {
            keylog_warn("Address count exceeds limit %d, ignoring extra addresses", MAX_ADDRESSES);
            break;
        }

        /* Decode address to 20-byte hash160 and insert into hash table */
        uint8_t h160[20];
        int ret = base58check_decode(line, h160);
        if (ret != 0) {
            keylog_warn("Address decode failed (ret=%d), skipping: %s", ret, line);
            skip_count++;
            continue;
        }
        ht_insert(h160);
        count++;
    }

    fclose(f);

    if (count == 0) {
        keylog_error("No valid addresses in file %s!", filename);
        return -1;
    }
    address_count = count;
    keylog_info("Successfully loaded %d addresses (skipped %d invalid)", count, skip_count);

    return 0;
}


int main(int argc, char *argv[])
{
    const char *address_file = NULL;
    int thread_count = 4;   /* default thread count */
    int opt;

#ifdef USE_GPU
#  define GETOPT_STR "a:n:gh"
#else
#  define GETOPT_STR "a:n:h"
#endif

    while ((opt = getopt(argc, argv, GETOPT_STR)) != -1) {
        switch (opt) {
        case 'a':
            address_file = optarg;
            break;
        case 'n': {
            int n = atoi(optarg);
            if (n <= 0) {
                fprintf(stderr, "Warning: invalid -n value (%s), using default thread count 4\n", optarg); /* before log_init, must use stderr */
                thread_count = 4;
            } else {
                thread_count = n;
            }
            break;
        }
#ifdef USE_GPU
        case 'g':
            use_gpu = 1;
            break;
#endif
        case 'h':
            fprintf(stdout, "Usage: ./keysearch -a <address_file> [-n <num_threads>] [-g] [-h]\n");
            fprintf(stdout, "  -a <address_file>  one target Bitcoin address per line (required)\n");
            fprintf(stdout, "  -n <num_threads>   number of worker threads, default 4 (CPU path)\n");
#ifdef USE_GPU
            fprintf(stdout, "  -g             enable Nvidia GPU acceleration path\n");
#endif
            fprintf(stdout, "  -h             show this help message\n");
            return 0;
        default:
            fprintf(stderr, "Error: unknown argument, use -h for help\n"); /* before log_init, must use stderr */
            return 1;
        }
    }

    if (!address_file) {
        fprintf(stderr, "Error: must specify address file with -a, use -h for help\n"); /* before log_init, must use stderr */
        return 1;
    }

    /* Initialize log file */
    if (log_init() != 0)
        return 1;

    /* Initialize hash table (open addressing, load factor <= 0.5, slot count = 2x address count rounded up to power of 2) */
    /* Initialize with max capacity: MAX_ADDRESSES * 2, rounded up to power of 2 */
    uint32_t ht_capacity = 1;
    while (ht_capacity < (uint32_t)MAX_ADDRESSES * 2)
        ht_capacity <<= 1;
    if (ht_init(ht_capacity) != 0) {
        keylog_error("Hash table memory allocation failed");
        log_close();
        return 1;
    }

    /* Load target addresses */
    if (load_target_addresses(address_file) != 0) {
        log_close();
        return 1;
    }

#ifdef USE_GPU
    if (use_gpu && thread_count != 4) {
        keylog_warn("GPU path: -n parameter is ignored");
    }
#endif

    keylog_info("Loaded %d target addresses, starting %d threads...",
                address_count, thread_count);

    /* Initialize secp256k1 context (SIGN used for public key creation) */
    secp_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!secp_ctx) {
        keylog_error("Failed to initialize secp256k1");
        log_close();
        return 1;
    }

#ifndef USE_PUBKEY_API_ONLY
    /* Initialize global generator G affine coordinates */
    if (keygen_init_generator(secp_ctx, &G_affine) != 0) {
        keylog_error("Failed to initialize generator G");
        secp256k1_context_destroy(secp_ctx);
        log_close();
        return 1;
    }
#endif

#ifdef USE_GPU
    if (use_gpu) {
        /* GPU path: initialize GPU, start search main loop */
        if (gpu_init(ht_slots, ht_mask + 1) != 0) {
            keylog_error("GPU initialization failed, exiting");
            secp256k1_context_destroy(secp_ctx);
            ht_free();
            log_close();
            return 1;
        }
        int gpu_ret = gpu_search();
        gpu_cleanup();
        secp256k1_context_destroy(secp_ctx);
        ht_free();
        log_close();
        return (gpu_ret == 0) ? 0 : 1;
    }
#endif

    /* CPU path: start threads */
    pthread_t *threads = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    struct thread_args *args = (struct thread_args *)malloc(thread_count * sizeof(struct thread_args));

    for (int i = 0; i < thread_count; i++) {
        args[i].thread_id = i + 1;
        pthread_create(&threads[i], NULL, search_key, &args[i]);
    }

    /* Wait for all threads to finish */
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    if (!found_flag) {
        keylog_info("All threads reached max attempts, no matching address found.");
    }

    /* Clean up resources */
    secp256k1_context_destroy(secp_ctx);
    ht_free();
    free(threads);
    free(args);

    log_close();
    return 0;
}

