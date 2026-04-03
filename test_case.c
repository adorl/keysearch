#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sha256.h"
#include "ripemd160.h"
#include "hash_utils.h"
#include "rand_key.h"
#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* GPU algorithm CPU-side simulation tests (defined in test_gpu.c) */
void run_gpu_tests(void);
/* secp256k1_keygen.h includes secp256k1.h (source directory version) in internal mode,
 * fallback mode requires system secp256k1.h, handled via conditional compilation */
#ifndef USE_PUBKEY_API_ONLY
#  include "secp256k1_keygen.h"
#else
#  include <secp256k1.h>
#  include "secp256k1_keygen.h"
#endif

/* ===================== Helper Functions ===================== */

static int pass_count = 0;
static int fail_count = 0;

/* secp256k1 context (read-only, thread-safe) */
secp256k1_context *secp_ctx = NULL;

#ifndef USE_PUBKEY_API_ONLY
/* Affine coordinates of global generator G (initialized by keygen_init_generator) */
secp256k1_ge G_affine;
#endif

/* Convert byte array to lowercase hex string (out must be at least len*2+1 bytes) */
static void bytes_to_hex_helper(const uint8_t *buf, size_t len, char *out) {
    for (size_t i = 0; i < len; i++) {
        sprintf(out + i * 2, "%02x", buf[i]);
    }
    out[len * 2] = '\0';
}

/* Assertion function: compare expected hex string with actual byte array */
static void check(const char *name, const char *expected_hex,
                  const uint8_t *actual, size_t len) {
    char actual_hex[len * 2 + 1];
    bytes_to_hex_helper(actual, len, actual_hex);
    if (strcmp(expected_hex, actual_hex) == 0) {
        printf("  [PASS] %s\n", name);
        pass_count++;
    } else {
        printf("  [FAIL] %s\n", name);
        printf("         expected: %s\n", expected_hex);
        printf("         actual:   %s\n", actual_hex);
        fail_count++;
    }
}

/* ===================== SHA256 Tests ===================== */

static void test_sha256(void) {
    printf("\n=== SHA256 Standard Test Vectors ===\n");

    sha256_ctx ctx;
    uint8_t digest[32];
    uint8_t buf[128];

    /* 1.1 empty string */
    sha256_init(&ctx);
    sha256_final(&ctx, digest);
    check("SHA256(\"\") empty string",
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          digest, 32);

    /* 1.2 "abc" */
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)"abc", 3);
    sha256_final(&ctx, digest);
    check("SHA256(\"abc\")",
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          digest, 32);

    /* 1.3 448-bit cross-block message */
    const char *msg448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)msg448, strlen(msg448));
    sha256_final(&ctx, digest);
    check("SHA256(448-bit cross-block message)",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          digest, 32);

    /* 1.4 55 bytes of 'a' */
    memset(buf, 'a', 55);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 55);
    sha256_final(&ctx, digest);
    check("SHA256('a'*55)",
          "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
          digest, 32);

    /* 1.5 56 bytes of 'a' */
    memset(buf, 'a', 56);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 56);
    sha256_final(&ctx, digest);
    check("SHA256('a'*56)",
          "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
          digest, 32);

    /* 1.6 64 bytes of 'a' */
    memset(buf, 'a', 64);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 64);
    sha256_final(&ctx, digest);
    check("SHA256('a'*64)",
          "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
          digest, 32);

    /* 1.7 65 bytes of 'a' */
    memset(buf, 'a', 65);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 65);
    sha256_final(&ctx, digest);
    check("SHA256('a'*65)",
          "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
          digest, 32);
}

/* ===================== RIPEMD160 Tests ===================== */

static void test_ripemd160(void) {
    printf("\n=== RIPEMD160 Standard Test Vectors ===\n");

    ripemd160_ctx ctx;
    uint8_t digest[20];
    uint8_t buf[128];

    /* 2.1 empty string */
    ripemd160_init(&ctx);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"\") empty string",
          "9c1185a5c5e9fc54612808977ee8f548b2258d31",
          digest, 20);

    /* 2.2 "abc" */
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)"abc", 3);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"abc\")",
          "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
          digest, 20);

    /* 2.3 26-byte alphabet */
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"abcdefghijklmnopqrstuvwxyz\")",
          "f71c27109c692c1b56bbdceb5b9d2865b3708dbc",
          digest, 20);

    /* 2.4 56-byte cross-block message */
    const char *msg56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)msg56, strlen(msg56));
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(56-byte cross-block message)",
          "12a053384a9c0c88e405a06c27dcf49ada62eb2b",
          digest, 20);

    /* 2.5 55 bytes of 'a' */
    memset(buf, 'a', 55);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 55);
    ripemd160_final(&ctx, digest);
    /* Self-consistency: compute reference value using ripemd160() convenience function */
    uint8_t ref[20];
    ripemd160(buf, 55, ref);
    char ref_hex[41];
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160('a'*55) self-consistency", ref_hex, digest, 20);

    /* 2.6 64 bytes of 'a' */
    memset(buf, 'a', 64);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 64);
    ripemd160_final(&ctx, digest);
    ripemd160(buf, 64, ref);
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160('a'*64) self-consistency", ref_hex, digest, 20);

    /* 2.7 32 zero bytes */
    memset(buf, 0x00, 32);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 32);
    ripemd160_final(&ctx, digest);
    ripemd160(buf, 32, ref);
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160(0x00*32) self-consistency", ref_hex, digest, 20);
}

/* ===================== Hash160 Combined Tests ===================== */

static void test_hash160(void) {
    printf("\n=== Hash160 Combined Scenario Verification ===\n");

    uint8_t sha_digest[32];
    uint8_t rmd_digest[20];

    /*
     * 3.1 Known 33-byte compressed pubkey (Bitcoin genesis block coinbase pubkey)
     *   pubkey: 04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb6
     *         49f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f
     *   compressed pubkey (33 bytes): 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   corresponding Hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
     */
    static const uint8_t compressed_pubkey[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17,
        0x98
    };
    sha256(compressed_pubkey, 33, sha_digest);
    ripemd160(sha_digest, 32, rmd_digest);
    check("Hash160(compressed pubkey 33 bytes)",
          "751e76e8199196d454941c45d1b3a323f1433bd6",
          rmd_digest, 20);

    /*
     * 3.2 Known 65-byte uncompressed pubkey (uncompressed form of the same key)
     *   uncompressed pubkey (65 bytes): 04 + X(32) + Y(32)
     *   corresponding Hash160: 91b24bf9f5288532960ac687abb035127b1d28a5
     */
    static const uint8_t uncompressed_pubkey[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };
    sha256(uncompressed_pubkey, 65, sha_digest);
    ripemd160(sha_digest, 32, rmd_digest);
    check("Hash160(uncompressed pubkey 65 bytes)",
          "91b24bf9f5288532960ac687abb035127b1d28a5",
          rmd_digest, 20);
}

/* ===================== Incremental Pubkey Derivation Tests ===================== */

/*
 * Helper: compute compressed pubkey serialized bytes directly from private key
 */
static void privkey_to_compressed_bytes(const uint8_t *privkey, uint8_t out[33])
{
    secp256k1_pubkey pubkey;
    size_t len = 33;
    secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
    secp256k1_ec_pubkey_serialize(secp_ctx, out, &len, &pubkey,
                                  SECP256K1_EC_COMPRESSED);
}

static void test_incremental_pubkey(void) {
    printf("\n=== Incremental Pubkey Derivation Correctness Tests ===\n");

    /* tweak = 1 (each step: privkey +1, pubkey point add G) */
    uint8_t tweak[32] = {0};
    tweak[31] = 1;

    /* ------------------------------------------------------------------ */
    /* 4.1  Single-step increment: k=1 -> k'=2
     *   privkey k  = 0x00...01  corresponding compressed pubkey P
     *   privkey k' = 0x00...02  corresponding compressed pubkey P' (computed directly)
     *   incremental derivation: P_incr = P + G (via secp256k1_ec_pubkey_tweak_add)
     *   expected: P' == P_incr
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey[32] = {0};
        privkey[31] = 1;  /* k = 1 */

        /* Compute pubkey for k'=2 directly */
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;
        uint8_t direct_bytes[33];
        privkey_to_compressed_bytes(privkey2, direct_bytes);

        /* Incremental derivation: add G to pubkey of k=1 */
        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
        secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);
        uint8_t incr_bytes[33];
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, incr_bytes, &len, &pubkey,
                                      SECP256K1_EC_COMPRESSED);

        char direct_hex[67];
        bytes_to_hex_helper(direct_bytes, 33, direct_hex);
        check("4.1 single-step incremental pubkey (k=1 -> k'=2) matches direct computation",
              direct_hex, incr_bytes, 33);
    }

    /* ------------------------------------------------------------------ */
    /* 4.2  hash160 consistency after single-step increment
     *   verify: pubkey_bytes_to_hash160(P_incr) == privkey_to_hash160(k')
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;  /* k' = 2 */

        /* Method A: privkey_to_hash160 computes directly from private key */
        uint8_t hash160_direct_comp[20];
        uint8_t hash160_direct_uncomp[20];
        privkey_to_hash160(privkey2, hash160_direct_comp, hash160_direct_uncomp);

        /* Method B: derive pubkey incrementally then use pubkey_bytes_to_hash160 */
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;
        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey1);
        secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);

        uint8_t comp_bytes[33];
        uint8_t uncomp_bytes[65];
        size_t len_c = 33, len_u = 65;
        secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes,   &len_c, &pubkey,
                                      SECP256K1_EC_COMPRESSED);
        secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bytes, &len_u, &pubkey,
                                      SECP256K1_EC_UNCOMPRESSED);

        uint8_t hash160_incr_comp[20];
        uint8_t hash160_incr_uncomp[20];
        pubkey_bytes_to_hash160(comp_bytes,   33, hash160_incr_comp);
        pubkey_bytes_to_hash160(uncomp_bytes, 65, hash160_incr_uncomp);

        /* Convert to hex for comparison */
        char direct_comp_hex[41];
        bytes_to_hex_helper(hash160_direct_comp, 20, direct_comp_hex);
        check("4.2a incremental compressed pubkey hash160 matches privkey_to_hash160 (k'=2)",
              direct_comp_hex, hash160_incr_comp, 20);

        char direct_uncomp_hex[41];
        bytes_to_hex_helper(hash160_direct_uncomp, 20, direct_uncomp_hex);
        check("4.2b incremental uncompressed pubkey hash160 matches privkey_to_hash160 (k'=2)",
              direct_uncomp_hex, hash160_incr_uncomp, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 4.3  Multi-step increment (10 steps): continuously derive from k=1 to k=11
     *   each step verifies incremental compressed pubkey hash160 matches direct computation
     * ------------------------------------------------------------------ */
    {
        printf("  [multi-step incremental derivation 10 steps, k=1..11]\n");
        uint8_t privkey[32] = {0};
        privkey[31] = 1;

        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);

        int all_pass = 1;
        for (int step = 1; step <= 10; step++) {
            /* Incremental derivation: privkey +1, pubkey point add G */
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak);
            secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);

            /* hash160 of incremental derivation */
            uint8_t comp_bytes[33];
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            uint8_t hash160_incr[20];
            pubkey_bytes_to_hash160(comp_bytes, 33, hash160_incr);

            /* hash160 computed directly */
            uint8_t hash160_direct[20];
            privkey_to_hash160(privkey, hash160_direct, NULL);

            if (memcmp(hash160_incr, hash160_direct, 20) != 0) {
                char incr_hex[41], direct_hex[41];
                bytes_to_hex_helper(hash160_incr,   20, incr_hex);
                bytes_to_hex_helper(hash160_direct, 20, direct_hex);
                printf("  [FAIL] step %d: incremental=%s direct=%s\n",
                       step, incr_hex, direct_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 4.3 multi-step incremental derivation hash160 (all 10 steps consistent)\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 4.4  Known vector: compressed pubkey hash160 for private key k=1
     *   compressed pubkey for private key k=1:
     *     0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   corresponding hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
     *   (same pubkey as used in test_hash160 3.1, cross-validation)
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        uint8_t hash160_comp[20];
        privkey_to_hash160(privkey1, hash160_comp, NULL);
        check("4.4 known vector: k=1 compressed pubkey hash160",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160_comp, 20);

        /* Also verify using pubkey_bytes_to_hash160 */
        uint8_t comp_bytes[33];
        privkey_to_compressed_bytes(privkey1, comp_bytes);
        uint8_t hash160_via_bytes[20];
        pubkey_bytes_to_hash160(comp_bytes, 33, hash160_via_bytes);
        check("4.4b pubkey_bytes_to_hash160(k=1 compressed pubkey) matches known vector",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160_via_bytes, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 4.5  BATCH_SIZE boundary: reset after 2048 continuous derivation steps, verify first step after reset
     *   base privkey k=0x00...05, derive 2048 steps to get k'=k+2048
     *   verify incremental hash160 matches hash160 computed directly for k'
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey[32] = {0};
        privkey[31] = 5;  /* k = 5 */

        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);

        /* Continuously derive 2048 steps */
        for (int i = 0; i < 2048; i++) {
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak);
            secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);
        }

        /* Incremental derivation result */
        uint8_t comp_bytes[33];
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes, &len, &pubkey,
                                      SECP256K1_EC_COMPRESSED);
        uint8_t hash160_incr[20];
        pubkey_bytes_to_hash160(comp_bytes, 33, hash160_incr);

        /* Compute hash160 for k+2048 directly */
        uint8_t hash160_direct[20];
        privkey_to_hash160(privkey, hash160_direct, NULL);

        char direct_hex[41];
        bytes_to_hex_helper(hash160_direct, 20, direct_hex);
        check("4.5 BATCH_SIZE(2048 steps) boundary: incremental hash160 matches direct computation",
              direct_hex, hash160_incr, 20);
    }
}

/* ===================== AVX2 Compression Function Tests ===================== */

#ifdef __AVX2__

/*
 * Helper: construct SHA256 padded block (single block, message < 56 bytes)
 *   SHA256 padding: message + 0x80 + zero padding + 8-byte big-endian message bit length
 *   block is big-endian uint32_t, used directly as sha256_compress input
 */
static void make_sha256_padded_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    /* Message bit length (big-endian) written to last 8 bytes */
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen >> 56);
    block[57] = (uint8_t)(bitlen >> 48);
    block[58] = (uint8_t)(bitlen >> 40);
    block[59] = (uint8_t)(bitlen >> 32);
    block[60] = (uint8_t)(bitlen >> 24);
    block[61] = (uint8_t)(bitlen >> 16);
    block[62] = (uint8_t)(bitlen >>  8);
    block[63] = (uint8_t)(bitlen      );
}

/*
 * Helper: construct RIPEMD160 padded block (single block, message < 56 bytes)
 *   RIPEMD160 padding: message + 0x80 + zero padding + 8-byte little-endian message bit length
 */
static void make_ripemd160_padded_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    /* Message bit length (little-endian) written to last 8 bytes */
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen      );
    block[57] = (uint8_t)(bitlen >>  8);
    block[58] = (uint8_t)(bitlen >> 16);
    block[59] = (uint8_t)(bitlen >> 24);
    block[60] = (uint8_t)(bitlen >> 32);
    block[61] = (uint8_t)(bitlen >> 40);
    block[62] = (uint8_t)(bitlen >> 48);
    block[63] = (uint8_t)(bitlen >> 56);
}

/* SHA256 initial state constants */
static const uint32_t SHA256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* RIPEMD160 initial state constants */
static const uint32_t RMD160_INIT[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/*
 * Helper: compute single-block compression result using scalar sha256_compress (via sha256_ctx internals)
 * Method: use full sha256_init + sha256_update + sha256_final flow;
 *         for single-block messages (< 56 bytes), state after final is the compression result.
 * Here we use sha256() convenience function to compute full hash as reference for AVX2 results.
 */

/*
 * Helper: convert state[8] to 32-byte big-endian digest (SHA256 output format)
 */
static void sha256_state_to_digest(const uint32_t state[8], uint8_t digest[32])
{
    for (int i = 0; i < 8; i++) {
        digest[i*4+0] = (uint8_t)(state[i] >> 24);
        digest[i*4+1] = (uint8_t)(state[i] >> 16);
        digest[i*4+2] = (uint8_t)(state[i] >>  8);
        digest[i*4+3] = (uint8_t)(state[i]      );
    }
}

/*
 * Helper: convert state[5] to 20-byte little-endian digest (RIPEMD160 output format)
 */
static void rmd160_state_to_digest(const uint32_t state[5], uint8_t digest[20])
{
    for (int i = 0; i < 5; i++) {
        digest[i*4+0] = (uint8_t)(state[i]      );
        digest[i*4+1] = (uint8_t)(state[i] >>  8);
        digest[i*4+2] = (uint8_t)(state[i] >> 16);
        digest[i*4+3] = (uint8_t)(state[i] >> 24);
    }
}

static void test_avx2_compress(void) {
    printf("\n=== AVX2 Compression Function Tests (sha256_compress_avx2 / ripemd160_compress_avx2) ===\n");

    /* ------------------------------------------------------------------ */
    /* 7.1  sha256_compress_avx2 — 8 lanes same message, result matches scalar sha256()
     *
     * Test method:
     *   - construct 8 identical padded blocks (for "abc")
     *   - compress with AVX2 function, get 8-lane state
     *   - convert state to digest, compare with sha256("abc") standard value
     * ------------------------------------------------------------------ */
    {
        /* padded block for "abc" */
        uint8_t block_abc[64];
        make_sha256_padded_block((const uint8_t *)"abc", 3, block_abc);

        /* 8-lane state, all initialized to SHA256 initial values */
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];
        for (int i = 0; i < 8; i++) {
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        sha256_compress_avx2(states, blocks);

        /* Convert each lane's state to digest, compare with standard value */
        uint8_t digest_avx2[32];
        for (int lane = 0; lane < 8; lane++) {
            sha256_state_to_digest(states_data[lane], digest_avx2);
            char name[64];
            snprintf(name, sizeof(name), "7.1 sha256_compress_avx2(\"abc\") lane%d matches standard value", lane);
            check(name,
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                  digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.2  sha256_compress_avx2 — 8 lanes different messages, each lane matches scalar sha256()
     *
     * 8 lane messages: empty string, "abc", 1 zero byte, 1 0xFF byte,
     *                   "hello", "world", ascending sequence (8 bytes), 'a'*10
     * ------------------------------------------------------------------ */
    {
        /* Prepare 8 different padded blocks */
        static const struct { const uint8_t *msg; size_t len; const char *expected; } cases[8] = {
            { (const uint8_t *)"",          0,  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
            { (const uint8_t *)"abc",       3,  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
            { (const uint8_t *)"\x00",      1,  "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d" },
            { (const uint8_t *)"\xff",      1,  "a8100ae6aa1940d0b663bb31cd466142ebbdbd5187131b92d93818987832eb89" },
            { (const uint8_t *)"hello",     5,  "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" },
            { (const uint8_t *)"world",     5,  "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7" },
            { (const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07", 8, "8a851ff82ee7048ad09ec3847f1ddf44944104d2cbd17ef4e3db22c6785a0d45" },
            { (const uint8_t *)"aaaaaaaaaa", 10, "bf2cb58a68f684d95a3b78ef8f661c9a4e5b09e82cc8f9cc88cce90528caeb27" },
        };

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_sha256_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx2(states, blocks);

        uint8_t digest_avx2[32];
        for (int i = 0; i < 8; i++) {
            sha256_state_to_digest(states_data[i], digest_avx2);
            char name[80];
            snprintf(name, sizeof(name), "7.2 sha256_compress_avx2 8 lanes different messages lane%d matches standard value", i);
            check(name, cases[i].expected, digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.3  sha256_compress_avx2 — cross-validation with scalar per lane (randomized state)
     *
     * Test method:
     *   - 8 lanes use different non-initial states (simulating intermediate state of multi-block message)
     *   - compress with AVX2 function, compare with scalar sha256_ctx internal compression result
     *   - obtain scalar reference by calling sha256_update twice (first block fixed, second block is test block)
     * ------------------------------------------------------------------ */
    {
        /* Use 8 different 33-byte messages (pubkey format) as first block content;
         * after sha256_update processes first block, ctx.state is the intermediate state;
         * then process second block (64 zero bytes) to get final state as reference */
        static const uint8_t first_msgs[8][33] = {
            { 0x02,0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,
              0x07,0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98 },
            { 0x03,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20 },
            { 0x02,0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,
              0xf0,0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0x03,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
              0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99 },
            { 0x02,0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,
              0x01,0x11,0x21,0x31,0x41,0x51,0x61,0x71,0x81,0x91,0xa1,0xb1,0xc1,0xd1,0xe1,0xf1,0x01 },
            { 0x03,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,
              0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
              0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01 },
            { 0x03,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe },
        };

        /* Second block: 64 zero bytes (as AVX2 compression input block) */
        uint8_t zero_block[64];
        memset(zero_block, 0, 64);

        /* Get intermediate state for each lane: after processing 33 bytes, ctx.buf has 33 bytes pending;
         * at this point state is still initial value, need to update 31 more bytes to fill a block and trigger compression.
         * Simpler approach: process 33+31=64 bytes (exactly one block) with sha256_ctx,
         * then read ctx.state as intermediate state.
         * But sha256_ctx's state field is internal, so we simulate by constructing a padded block.
         *
         * Actually, the simplest cross-validation method:
         *   for each lane i, construct padded block = internal block of sha256_33(first_msgs[i]),
         *   i.e.: first_msgs[i] (33 bytes) + 0x80 + 22 zero bytes + 8-byte bit length (264 bits)
         *   this is exactly the first (and only) block processed internally by sha256_33.
         *   AVX2 compressed state converted to digest should equal sha256_33(first_msgs[i]).
         */
        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_sha256_padded_block(first_msgs[i], 33, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx2(states, blocks);

        /* Reference values: output of sha256_33() */
        uint8_t digest_avx2[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 8; i++) {
            sha256_state_to_digest(states_data[i], digest_avx2);
            sha256_33(first_msgs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.3 sha256_compress_avx2 8-lane 33-byte pubkey lane%d matches sha256_33", i);
            check(name, ref_hex, digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.4  ripemd160_compress_avx2 — 8 lanes same message, result matches scalar ripemd160()
     * ------------------------------------------------------------------ */
    {
        /* RIPEMD160 padded block for "abc" */
        uint8_t block_abc[64];
        make_ripemd160_padded_block((const uint8_t *)"abc", 3, block_abc);

        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];
        for (int i = 0; i < 8; i++) {
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        for (int lane = 0; lane < 8; lane++) {
            rmd160_state_to_digest(states_data[lane], digest_avx2);
            char name[64];
            snprintf(name, sizeof(name), "7.4 ripemd160_compress_avx2(\"abc\") lane%d matches standard value", lane);
            check(name,
                  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
                  digest_avx2, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.5  ripemd160_compress_avx2 — 8 lanes different messages, each lane matches scalar
     * ------------------------------------------------------------------ */
    {
        static const struct { const uint8_t *msg; size_t len; const char *expected; } cases[8] = {
            { (const uint8_t *)"",          0,  "9c1185a5c5e9fc54612808977ee8f548b2258d31" },
            { (const uint8_t *)"abc",       3,  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc" },
            { (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26, "f71c27109c692c1b56bbdceb5b9d2865b3708dbc" },
            { (const uint8_t *)"hello",     5,  "108f07b8382412612c048d07d13f814118445acd" },
            { (const uint8_t *)"world",     5,  "9b2a277a3e3b3a31b3114ca2d73be6d493d037f9" },
            { (const uint8_t *)"\x00",      1,  "c81b94933420221a7ac004a90242d8b1d3e5070d" },
            { (const uint8_t *)"\xff",      1,  "f7f5b1e2d1b9e3b3e3b3e3b3e3b3e3b3e3b3e3b3" /* placeholder, self-consistency used below */ },
            { (const uint8_t *)"0123456789", 10, "9a1c58e8f2f9b3e3b3e3b3e3b3e3b3e3b3e3b3e3" /* placeholder */ },
        };
        /* Note: some expected values use placeholders, actual verification uses self-consistency (compared with generic ripemd160) */

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_ripemd160_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 8; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx2);
            /* Compute reference value using generic interface (self-consistency, no hardcoded expected values) */
            ripemd160(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.5 ripemd160_compress_avx2 8 lanes different messages lane%d matches generic ripemd160", i);
            check(name, ref_hex, digest_avx2, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.6  ripemd160_compress_avx2 — 8 lanes 32-byte messages (ripemd160_32 scenario)
     *
     * Test method:
     *   - 8 lanes use different 32-byte inputs (simulating sha256 output)
     *   - AVX2 compression result compared with ripemd160_32()
     * ------------------------------------------------------------------ */
    {
        static const uint8_t inputs[8][32] = {
            /* sha256(G_compressed) */
            { 0x0b,0x7c,0x28,0xc9,0xb7,0x29,0x0c,0x98,0xd7,0x43,0x8e,0x70,0xb3,0xd3,0xf7,0xc8,
              0x48,0xfb,0xd7,0xd1,0xdc,0x19,0x4f,0xf8,0x3f,0x4f,0x7c,0xc9,0xb1,0x37,0x8e,0x98 },
            /* all zeros */
            { 0 },
            /* all 0xFF */
            { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
            /* ascending 0x00~0x1F */
            { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f },
            /* descending 0xFF~0xE0 */
            { 0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,
              0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            /* alternating 0xAA/0x55 */
            { 0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,
              0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55 },
            /* random sample 1 */
            { 0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
              0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            /* random sample 2 */
            { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,
              0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff },
        };

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_ripemd160_padded_block(inputs[i], 32, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 8; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx2);
            ripemd160_32(inputs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.6 ripemd160_compress_avx2 8-lane 32-byte input lane%d matches ripemd160_32", i);
            check(name, ref_hex, digest_avx2, 20);
        }
    }
}

/* ===================== hash160_8way Tests ===================== */

/*
 * Test hash160_8way_compressed and hash160_8way_uncompressed
 * Verify 8-way parallel results are consistent with scalar pubkey_bytes_to_hash160
 */
static void test_hash160_8way(void) {
    printf("\n=== hash160_8way 8-way parallel hash160 tests ===\n");

    /* Known pubkeys: compressed and uncompressed pubkey of G point (private key k=1) */
    static const uint8_t G_comp[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
    };
    static const uint8_t G_uncomp[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };

    /* ------------------------------------------------------------------ */
    /* 8.1  hash160_8way_compressed — 8 identical inputs (G point compressed pubkey)
     *      Verify all 8 lane outputs are consistent with scalar pubkey_bytes_to_hash160
     * ------------------------------------------------------------------ */
    {
        const uint8_t *comp_ptrs[8];
        for (int i = 0; i < 8; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        /* scalar reference value */
        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_comp, 33, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 8; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "8.1 hash160_8way_compressed(G point) lane%d consistent with scalar", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.2  hash160_8way_uncompressed — 8 identical inputs (G point uncompressed pubkey)
     *      Verify all 8 lane outputs are consistent with scalar pubkey_bytes_to_hash160
     * ------------------------------------------------------------------ */
    {
        const uint8_t *uncomp_ptrs[8];
        for (int i = 0; i < 8; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        /* scalar reference value */
        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_uncomp, 65, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 8; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "8.2 hash160_8way_uncompressed(G point) lane%d consistent with scalar", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.3  hash160_8way_compressed — 8 different inputs (compressed pubkeys for k=1..8)
     *      Cross-validate each lane with scalar privkey_to_hash160
     * ------------------------------------------------------------------ */
    {
        /* Build compressed pubkeys for k=1..8 */
        uint8_t comp_bufs[8][33];
        const uint8_t *comp_ptrs[8];
        uint8_t ref_hash160s[8][20];

        for (int i = 0; i < 8; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);
            privkey_to_compressed_bytes(privkey, comp_bufs[i]);
            comp_ptrs[i] = comp_bufs[i];
            privkey_to_hash160(privkey, ref_hash160s[i], NULL);
        }

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        for (int lane = 0; lane < 8; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "8.3 hash160_8way_compressed(k=%d) lane%d consistent with scalar", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.4  hash160_8way_uncompressed — 8 different inputs (uncompressed pubkeys for k=1..8)
     *      Cross-validate each lane with scalar privkey_to_hash160
     * ------------------------------------------------------------------ */
    {
        /* Build uncompressed pubkeys for k=1..8 */
        uint8_t uncomp_bufs[8][65];
        const uint8_t *uncomp_ptrs[8];
        uint8_t ref_hash160s[8][20];

        for (int i = 0; i < 8; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);

            secp256k1_pubkey pubkey;
            size_t len = 65;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
            secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bufs[i], &len,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);
            uncomp_ptrs[i] = uncomp_bufs[i];
            privkey_to_hash160(privkey, NULL, ref_hash160s[i]);
        }

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        for (int lane = 0; lane < 8; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "8.4 hash160_8way_uncompressed(k=%d) lane%d consistent with scalar", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.5  Known vector: hash160_8way_compressed(G point) lane0 consistent with known hash160
     *      G point compressed pubkey hash160 = 751e76e8199196d454941c45d1b3a323f1433bd6
     * ------------------------------------------------------------------ */
    {
        const uint8_t *comp_ptrs[8];
        for (int i = 0; i < 8; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        check("8.5 hash160_8way_compressed(G point) consistent with known vector",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[0], 20);
    }

    /* ------------------------------------------------------------------ */
    /* 8.6  Known vector: hash160_8way_uncompressed(G point) lane0 consistent with known hash160
     *      G point uncompressed pubkey hash160 = 91b24bf9f5288532960ac687abb035127b1d28a5
     * ------------------------------------------------------------------ */
    {
        const uint8_t *uncomp_ptrs[8];
        for (int i = 0; i < 8; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        check("8.6 hash160_8way_uncompressed(G point) consistent with known vector",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[0], 20);
    }
}

#endif /* __AVX2__ */

/* ===================== AVX-512 Compression Function Tests ===================== */

#ifdef __AVX512F__

__attribute__((target("avx512f")))
/*
 * Helper: extract per-lane SHA256 digest from SoA state.
 * soa_state[8] holds 16 lanes; this extracts lane `lane` into digest[32].
 */
static void sha256_soa_state_to_digest(const __m512i soa_state[8], int lane, uint8_t digest[32])
{
    uint32_t tmp[16] __attribute__((aligned(64)));
    for (int w = 0; w < 8; w++) {
        _mm512_store_si512((__m512i *)tmp, soa_state[w]);
        uint32_t val = tmp[lane];
        digest[w*4+0] = (uint8_t)(val >> 24);
        digest[w*4+1] = (uint8_t)(val >> 16);
        digest[w*4+2] = (uint8_t)(val >>  8);
        digest[w*4+3] = (uint8_t)(val      );
    }
}

/*
 * Helper: extract per-lane RIPEMD160 digest from SoA state.
 * soa_state[5] holds 16 lanes; this extracts lane `lane` into digest[20].
 */
static void rmd160_soa_state_to_digest(const __m512i soa_state[5], int lane, uint8_t digest[20])
{
    uint32_t tmp[16] __attribute__((aligned(64)));
    for (int w = 0; w < 5; w++) {
        _mm512_store_si512((__m512i *)tmp, soa_state[w]);
        uint32_t val = tmp[lane];
        /* RIPEMD160 is little-endian */
        digest[w*4+0] = (uint8_t)(val      );
        digest[w*4+1] = (uint8_t)(val >>  8);
        digest[w*4+2] = (uint8_t)(val >> 16);
        digest[w*4+3] = (uint8_t)(val >> 24);
    }
}

/*
 * Helper: load 16 RIPEMD160 message words from 16 padded blocks into __m512i[16].
 * blocks[16] are pointers to 64-byte padded blocks (little-endian uint32).
 */
static void load_rmd160_words_16way(const uint8_t *blocks[16], __m512i w[16])
{
    for (int i = 0; i < 16; i++) {
        const int off = i * 4;
        uint32_t lanes[16] __attribute__((aligned(64)));
        for (int k = 0; k < 16; k++) {
            uint32_t v;
            __builtin_memcpy(&v, blocks[k] + off, sizeof(v));
            lanes[k] = v;
        }
        w[i] = _mm512_load_si512((const void *)lanes);
    }
}

static void test_avx512_compress(void) {
    printf("\n=== AVX-512 compression function tests (sha256_compress_avx512_soa / ripemd160_compress_avx512_soa) ===\n");

    /* ------------------------------------------------------------------ */
    /* 10.1  sha256_compress_avx512_soa — 16 identical messages, results consistent with scalar sha256() */
    {
        uint8_t block_abc[64];
        make_sha256_padded_block((const uint8_t *)"abc", 3, block_abc);

        const uint8_t *blocks[16];
        for (int i = 0; i < 16; i++) {
            blocks[i] = block_abc;
        }

        __m512i soa_state[8];
        soa_state[0] = _mm512_set1_epi32(0x6a09e667);
        soa_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
        soa_state[2] = _mm512_set1_epi32(0x3c6ef372);
        soa_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
        soa_state[4] = _mm512_set1_epi32(0x510e527f);
        soa_state[5] = _mm512_set1_epi32((int)0x9b05688c);
        soa_state[6] = _mm512_set1_epi32(0x1f83d9ab);
        soa_state[7] = _mm512_set1_epi32(0x5be0cd19);

        sha256_compress_avx512_soa(soa_state, blocks);

        uint8_t digest_avx512[32];
        for (int lane = 0; lane < 16; lane++) {
            sha256_soa_state_to_digest(soa_state, lane, digest_avx512);
            char name[80];
            snprintf(name, sizeof(name), "10.1 sha256_compress_avx512_soa(\"abc\") lane%d consistent with standard value", lane);
            check(name,
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                  digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.2  sha256_compress_avx512_soa — 16 different messages, each lane consistent with scalar sha256() */
    {
        static const struct { const uint8_t *msg; size_t len; } cases[16] = {
            { (const uint8_t *)"",           0  },
            { (const uint8_t *)"abc",        3  },
            { (const uint8_t *)"\x00",       1  },
            { (const uint8_t *)"\xff",       1  },
            { (const uint8_t *)"hello",      5  },
            { (const uint8_t *)"world",      5  },
            { (const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07", 8 },
            { (const uint8_t *)"aaaaaaaaaa", 10 },
            { (const uint8_t *)"bitcoin",    7  },
            { (const uint8_t *)"secp256k1",  9  },
            { (const uint8_t *)"ripemd160",  9  },
            { (const uint8_t *)"sha256",     6  },
            { (const uint8_t *)"avx512",     6  },
            { (const uint8_t *)"keysearch",  9  },
            { (const uint8_t *)"test16way",  9  },
            { (const uint8_t *)"lane15msg",  9  },
        };

        uint8_t padded_blocks[16][64];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_sha256_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            blocks[i] = padded_blocks[i];
        }

        __m512i soa_state[8];
        soa_state[0] = _mm512_set1_epi32(0x6a09e667);
        soa_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
        soa_state[2] = _mm512_set1_epi32(0x3c6ef372);
        soa_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
        soa_state[4] = _mm512_set1_epi32(0x510e527f);
        soa_state[5] = _mm512_set1_epi32((int)0x9b05688c);
        soa_state[6] = _mm512_set1_epi32(0x1f83d9ab);
        soa_state[7] = _mm512_set1_epi32(0x5be0cd19);

        sha256_compress_avx512_soa(soa_state, blocks);

        uint8_t digest_avx512[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 16; i++) {
            sha256_soa_state_to_digest(soa_state, i, digest_avx512);
            sha256(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.2 sha256_compress_avx512_soa 16-way different messages lane%d consistent with scalar", i);
            check(name, ref_hex, digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.3  sha256_compress_avx512_soa — 16-way 33-byte pubkeys, cross-validate with sha256_33 */
    {
        static const uint8_t msgs[16][33] = {
            { 0x02,0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,
              0x07,0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98 },
            { 0x03,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20 },
            { 0x02,0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,
              0xf0,0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0x03,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
              0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99 },
            { 0x02,0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,
              0x01,0x11,0x21,0x31,0x41,0x51,0x61,0x71,0x81,0x91,0xa1,0xb1,0xc1,0xd1,0xe1,0xf1,0x01 },
            { 0x03,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,
              0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
              0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01 },
            { 0x03,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe },
            { 0x02,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,
              0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00 },
            { 0x03,0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
              0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            { 0x02,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,0x4a,0x5b,0x6c,0x7d,0x8e,
              0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f },
            { 0x03,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,
              0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0 },
            { 0x02,0x55,0x44,0x33,0x22,0x11,0x00,0xff,0xee,0xdd,0xcc,0xbb,0xaa,0x99,0x88,0x77,
              0x66,0x55,0x44,0x33,0x22,0x11,0x00,0xff,0xee,0xdd,0xcc,0xbb,0xaa,0x99,0x88,0x77,0x66 },
            { 0x03,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
              0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef },
            { 0x02,0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b,0x3c,0x2d,0x1e,
              0x0f,0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f },
            { 0x03,0x80,0x70,0x60,0x50,0x40,0x30,0x20,0x10,0x08,0x07,0x06,0x05,0x04,0x03,0x02,
              0x01,0x80,0x70,0x60,0x50,0x40,0x30,0x20,0x10,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01 },
        };

        uint8_t padded_blocks[16][64];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_sha256_padded_block(msgs[i], 33, padded_blocks[i]);
            blocks[i] = padded_blocks[i];
        }

        __m512i soa_state[8];
        soa_state[0] = _mm512_set1_epi32(0x6a09e667);
        soa_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
        soa_state[2] = _mm512_set1_epi32(0x3c6ef372);
        soa_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
        soa_state[4] = _mm512_set1_epi32(0x510e527f);
        soa_state[5] = _mm512_set1_epi32((int)0x9b05688c);
        soa_state[6] = _mm512_set1_epi32(0x1f83d9ab);
        soa_state[7] = _mm512_set1_epi32(0x5be0cd19);

        sha256_compress_avx512_soa(soa_state, blocks);

        uint8_t digest_avx512[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 16; i++) {
            sha256_soa_state_to_digest(soa_state, i, digest_avx512);
            sha256_33(msgs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.3 sha256_compress_avx512_soa 16-way 33-byte pubkey lane%d consistent with sha256_33", i);
            check(name, ref_hex, digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.4  ripemd160_compress_avx512_soa — 16 identical messages, results consistent with scalar ripemd160() */
    {
        uint8_t block_abc[64];
        make_ripemd160_padded_block((const uint8_t *)"abc", 3, block_abc);

        const uint8_t *blocks[16];
        for (int i = 0; i < 16; i++) {
            blocks[i] = block_abc;
        }

        __m512i rmd_w[16];
        load_rmd160_words_16way(blocks, rmd_w);

        __m512i soa_state[5];
        soa_state[0] = _mm512_set1_epi32(0x67452301);
        soa_state[1] = _mm512_set1_epi32((int)0xEFCDAB89);
        soa_state[2] = _mm512_set1_epi32((int)0x98BADCFE);
        soa_state[3] = _mm512_set1_epi32(0x10325476);
        soa_state[4] = _mm512_set1_epi32((int)0xC3D2E1F0);

        ripemd160_compress_avx512_soa(soa_state, rmd_w);

        uint8_t digest_avx512[20];
        for (int lane = 0; lane < 16; lane++) {
            rmd160_soa_state_to_digest(soa_state, lane, digest_avx512);
            char name[80];
            snprintf(name, sizeof(name), "10.4 ripemd160_compress_avx512_soa(\"abc\") lane%d consistent with standard value", lane);
            check(name,
                  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
                  digest_avx512, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.5  ripemd160_compress_avx512_soa — 16 different messages, each lane consistent with scalar */
    {
        static const struct { const uint8_t *msg; size_t len; } cases[16] = {
            { (const uint8_t *)"",           0  },
            { (const uint8_t *)"abc",        3  },
            { (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26 },
            { (const uint8_t *)"hello",      5  },
            { (const uint8_t *)"world",      5  },
            { (const uint8_t *)"\x00",       1  },
            { (const uint8_t *)"\xff",       1  },
            { (const uint8_t *)"0123456789", 10 },
            { (const uint8_t *)"bitcoin",    7  },
            { (const uint8_t *)"secp256k1",  9  },
            { (const uint8_t *)"ripemd160",  9  },
            { (const uint8_t *)"sha256",     6  },
            { (const uint8_t *)"avx512",     6  },
            { (const uint8_t *)"keysearch",  9  },
            { (const uint8_t *)"test16way",  9  },
            { (const uint8_t *)"lane15msg",  9  },
        };

        uint8_t padded_blocks[16][64];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_ripemd160_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            blocks[i] = padded_blocks[i];
        }

        __m512i rmd_w[16];
        load_rmd160_words_16way(blocks, rmd_w);

        __m512i soa_state[5];
        soa_state[0] = _mm512_set1_epi32(0x67452301);
        soa_state[1] = _mm512_set1_epi32((int)0xEFCDAB89);
        soa_state[2] = _mm512_set1_epi32((int)0x98BADCFE);
        soa_state[3] = _mm512_set1_epi32(0x10325476);
        soa_state[4] = _mm512_set1_epi32((int)0xC3D2E1F0);

        ripemd160_compress_avx512_soa(soa_state, rmd_w);

        uint8_t digest_avx512[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 16; i++) {
            rmd160_soa_state_to_digest(soa_state, i, digest_avx512);
            ripemd160(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.5 ripemd160_compress_avx512_soa 16-way different messages lane%d consistent with generic ripemd160", i);
            check(name, ref_hex, digest_avx512, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.6  ripemd160_compress_avx512_soa — 16-way 32-byte messages (ripemd160_32 scenario) */
    {
        static const uint8_t inputs[16][32] = {
            { 0x0b,0x7c,0x28,0xc9,0xb7,0x29,0x0c,0x98,0xd7,0x43,0x8e,0x70,0xb3,0xd3,0xf7,0xc8,
              0x48,0xfb,0xd7,0xd1,0xdc,0x19,0x4f,0xf8,0x3f,0x4f,0x7c,0xc9,0xb1,0x37,0x8e,0x98 },
            { 0 },
            { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
            { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f },
            { 0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,
              0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,
              0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55 },
            { 0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
              0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,
              0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff },
            { 0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,
              0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00 },
            { 0xa1,0xb2,0xc3,0xd4,0xe5,0xf6,0x07,0x18,0x29,0x3a,0x4b,0x5c,0x6d,0x7e,0x8f,0x90,
              0xa1,0xb2,0xc3,0xd4,0xe5,0xf6,0x07,0x18,0x29,0x3a,0x4b,0x5c,0x6d,0x7e,0x8f,0x90 },
            { 0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,0x01,
              0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,0x01 },
            { 0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,
              0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,
              0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0x9b,0x9c,0x9d,0x9e,0x9f },
            { 0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x7b,0x7c,0x7d,0x7e,0x7f,
              0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x6b,0x6c,0x6d,0x6e,0x6f },
            { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
              0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80 },
            { 0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,
              0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f },
        };

        uint8_t padded_blocks[16][64];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_ripemd160_padded_block(inputs[i], 32, padded_blocks[i]);
            blocks[i] = padded_blocks[i];
        }

        __m512i rmd_w[16];
        load_rmd160_words_16way(blocks, rmd_w);

        __m512i soa_state[5];
        soa_state[0] = _mm512_set1_epi32(0x67452301);
        soa_state[1] = _mm512_set1_epi32((int)0xEFCDAB89);
        soa_state[2] = _mm512_set1_epi32((int)0x98BADCFE);
        soa_state[3] = _mm512_set1_epi32(0x10325476);
        soa_state[4] = _mm512_set1_epi32((int)0xC3D2E1F0);

        ripemd160_compress_avx512_soa(soa_state, rmd_w);

        uint8_t digest_avx512[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 16; i++) {
            rmd160_soa_state_to_digest(soa_state, i, digest_avx512);
            ripemd160_32(inputs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.6 ripemd160_compress_avx512_soa 16-way 32-byte input lane%d consistent with ripemd160_32", i);
            check(name, ref_hex, digest_avx512, 20);
        }
    }
}

/* ===================== hash160_16way Tests ===================== */

__attribute__((target("avx512f")))
static void test_hash160_16way(void) {
    printf("\n=== hash160_16way 16-way parallel hash160 tests ===\n");

    static const uint8_t G_comp[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
    };
    static const uint8_t G_uncomp[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };

    /* ------------------------------------------------------------------ */
    /* 11.1  hash160_16way_compressed_prepadded — 16 identical inputs (G point compressed pubkey) */
    {
        uint8_t padded_buf[64];
        memcpy(padded_buf, G_comp, 33);
        sha256_pad_block_33(padded_buf);
        const uint8_t *comp_ptrs[16];
        for (int i = 0; i < 16; i++) comp_ptrs[i] = padded_buf;

        uint8_t hash160s[16][20];
        hash160_16way_compressed_prepadded(comp_ptrs, hash160s);

        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_comp, 33, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 16; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "11.1 hash160_16way_compressed_prepadded(G point) lane%d consistent with scalar", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.2  hash160_16way_uncompressed_prepadded — 16 identical inputs (G point uncompressed pubkey) */
    {
        uint8_t padded_buf[128];
        memcpy(padded_buf, G_uncomp, 65);
        sha256_pad_block2_65(padded_buf);
        const uint8_t *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) uncomp_ptrs[i] = padded_buf;

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed_prepadded(uncomp_ptrs, hash160s);

        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_uncomp, 65, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 16; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "11.2 hash160_16way_uncompressed_prepadded(G point) lane%d consistent with scalar", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.3  hash160_16way_compressed_prepadded — 16 different inputs (compressed pubkeys for k=1..16) */
    {
        uint8_t comp_bufs[16][64];
        const uint8_t *comp_ptrs[16];
        uint8_t ref_hash160s[16][20];

        for (int i = 0; i < 16; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);
            uint8_t pubkey_raw[33];
            privkey_to_compressed_bytes(privkey, pubkey_raw);
            memcpy(comp_bufs[i], pubkey_raw, 33);
            sha256_pad_block_33(comp_bufs[i]);
            comp_ptrs[i] = comp_bufs[i];
            privkey_to_hash160(privkey, ref_hash160s[i], NULL);
        }

        uint8_t hash160s[16][20];
        hash160_16way_compressed_prepadded(comp_ptrs, hash160s);

        for (int lane = 0; lane < 16; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "11.3 hash160_16way_compressed_prepadded(k=%d) lane%d consistent with scalar", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.4  hash160_16way_uncompressed_prepadded — 16 different inputs (uncompressed pubkeys for k=1..16) */
    {
        uint8_t uncomp_bufs[16][128];
        const uint8_t *uncomp_ptrs[16];
        uint8_t ref_hash160s[16][20];

        for (int i = 0; i < 16; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);

            secp256k1_pubkey pubkey;
            size_t len = 65;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
            secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bufs[i], &len,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);
            sha256_pad_block2_65(uncomp_bufs[i]);
            uncomp_ptrs[i] = uncomp_bufs[i];
            privkey_to_hash160(privkey, NULL, ref_hash160s[i]);
        }

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed_prepadded(uncomp_ptrs, hash160s);

        for (int lane = 0; lane < 16; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "11.4 hash160_16way_uncompressed_prepadded(k=%d) lane%d consistent with scalar", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.5  Known vector: hash160_16way_compressed_prepadded(G point) lane0 consistent with known hash160 */
    {
        uint8_t padded_buf[64];
        memcpy(padded_buf, G_comp, 33);
        sha256_pad_block_33(padded_buf);
        const uint8_t *comp_ptrs[16];
        for (int i = 0; i < 16; i++) comp_ptrs[i] = padded_buf;

        uint8_t hash160s[16][20];
        hash160_16way_compressed_prepadded(comp_ptrs, hash160s);

        check("11.5 hash160_16way_compressed_prepadded(G point) consistent with known vector",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[0], 20);
        check("11.5b hash160_16way_compressed_prepadded(G point) lane15 consistent with known vector",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[15], 20);
    }

    /* ------------------------------------------------------------------ */
    /* 11.6  Known vector: hash160_16way_uncompressed_prepadded(G point) lane0 consistent with known hash160 */
    {
        uint8_t padded_buf[128];
        memcpy(padded_buf, G_uncomp, 65);
        sha256_pad_block2_65(padded_buf);
        const uint8_t *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) uncomp_ptrs[i] = padded_buf;

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed_prepadded(uncomp_ptrs, hash160s);

        check("11.6 hash160_16way_uncompressed_prepadded(G point) consistent with known vector",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[0], 20);
        check("11.6b hash160_16way_uncompressed_prepadded(G point) lane15 consistent with known vector",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[15], 20);
    }
}

/* ===================== ht_contains_16way Tests ===================== */

__attribute__((target("avx512f")))
static void test_ht_contains_16way_func(void) {
    printf("\n=== ht_contains_16way AVX-512 batch lookup tests ===\n");

    if (ht_init(128) != 0) {
        printf("  [FAIL] 12.0 ht_init failed\n");
        fail_count++;
        return;
    }

    /* Prepare 16 known hash160 values */
    uint8_t known[16][20];
    for (int i = 0; i < 16; i++) {
        memset(known[i], 0, 20);
        known[i][0] = (uint8_t)(0x10 + i);
        known[i][1] = (uint8_t)(0x20 + i);
        known[i][2] = (uint8_t)(0x30 + i);
        known[i][3] = (uint8_t)(0x40 + i);
        known[i][19] = (uint8_t)(i + 1);
        ht_insert(known[i]);
    }

    /* Prepare 16 unknown (not inserted) hash160 values */
    uint8_t unknown[16][20];
    for (int i = 0; i < 16; i++) {
        memset(unknown[i], 0, 20);
        unknown[i][0] = (uint8_t)(0xf0 + i);
        unknown[i][1] = (uint8_t)(0xe0 + i);
        unknown[i][2] = (uint8_t)(0xd0 + i);
        unknown[i][3] = (uint8_t)(0xc0 + i);
        unknown[i][19] = (uint8_t)(i + 0x80);
    }

    /* 12.1 all 16 lanes hit */
    const uint8_t *ptrs_all[16];
    for (int i = 0; i < 16; i++) ptrs_all[i] = known[i];
    uint16_t mask = ht_contains_16way(ptrs_all);
    if (mask == 0xffff) {
        printf("  [PASS] 12.1 all 16 lanes hit, mask=0xffff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.1 all 16 lanes hit, expected mask=0xffff, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.2 all 16 lanes miss */
    const uint8_t *ptrs_none[16];
    for (int i = 0; i < 16; i++) ptrs_none[i] = unknown[i];
    mask = ht_contains_16way(ptrs_none);
    if (mask == 0x0000) {
        printf("  [PASS] 12.2 all 16 lanes miss, mask=0x0000\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.2 all 16 lanes miss, expected mask=0x0000, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.3 partial hit: even lanes hit (0,2,4,6,8,10,12,14), expected mask=0x5555 */
    const uint8_t *ptrs_mix[16];
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i % 2 == 0) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x5555) {
        printf("  [PASS] 12.3 partial hit (even lanes), mask=0x5555\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.3 partial hit (even lanes), expected mask=0x5555, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.4 partial hit: odd lanes hit (1,3,5,7,9,11,13,15), expected mask=0xaaaa */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i % 2 == 1) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0xaaaa) {
        printf("  [PASS] 12.4 partial hit (odd lanes), mask=0xaaaa\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.4 partial hit (odd lanes), expected mask=0xaaaa, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.5 only lane0 hits, expected mask=0x0001 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i == 0) ? known[0] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x0001) {
        printf("  [PASS] 12.5 only lane0 hits, mask=0x0001\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.5 only lane0 hits, expected mask=0x0001, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.6 only lane15 hits, expected mask=0x8000 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i == 15) ? known[15] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x8000) {
        printf("  [PASS] 12.6 only lane15 hits, mask=0x8000\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.6 only lane15 hits, expected mask=0x8000, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.7 first 8 lanes hit, last 8 miss, expected mask=0x00ff */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i < 8) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x00ff) {
        printf("  [PASS] 12.7 first 8 lanes hit, mask=0x00ff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.7 first 8 lanes hit, expected mask=0x00ff, actual=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.8 last 8 lanes hit, first 8 miss, expected mask=0xff00 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i >= 8) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0xff00) {
        printf("  [PASS] 12.8 last 8 lanes hit, mask=0xff00\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.8 last 8 lanes hit, expected mask=0xff00, actual=0x%04x\n", mask);
        fail_count++;
    }

    ht_free();
}

#endif /* __AVX512F__ */

/* ===================== Open-addressing Hash Table Tests ===================== */

static void test_ht_openaddr(void) {
    printf("\n=== Open-addressing Hash Table Tests ===\n");

    /* Initialize hash table (16 slots, sufficient for testing) */
    if (ht_init(16) != 0) {
        printf("  [FAIL] 8.0 ht_init failed\n");
        fail_count++;
        return;
    }

    /* 8.1 Insert known hash160, verify ht_contains returns 1 */
    uint8_t h1[20] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,
                      0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14};
    uint8_t h2[20] = {0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,
                      0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd};
    ht_insert(h1);
    ht_insert(h2);

    if (ht_contains(h1) == 1) {
        printf("  [PASS] 8.1 ht_contains(h1) hit\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.1 ht_contains(h1) should hit but missed\n");
        fail_count++;
    }

    if (ht_contains(h2) == 1) {
        printf("  [PASS] 8.2 ht_contains(h2) hit\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.2 ht_contains(h2) should hit but missed\n");
        fail_count++;
    }

    /* 8.3 Look up hash160 not inserted, verify returns 0 */
    uint8_t h3[20] = {0xde,0xad,0xbe,0xef,0x00,0x00,0x00,0x00,0x00,0x00,
                      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    if (ht_contains(h3) == 0) {
        printf("  [PASS] 8.3 ht_contains(h3) miss (correct)\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.3 ht_contains(h3) should miss but hit\n");
        fail_count++;
    }

    /* 8.4 fp==0 boundary: first 4 bytes of hash160 all zero */
    uint8_t h_fp0[20] = {0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,
                         0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    ht_insert(h_fp0);
    if (ht_contains(h_fp0) == 1) {
        printf("  [PASS] 8.4 fp==0 boundary: ht_contains(h_fp0) hit\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.4 fp==0 boundary: ht_contains(h_fp0) should hit but missed\n");
        fail_count++;
    }

    /* 8.5 Bulk insert (load factor near 0.5), reinitialize with larger table */
    ht_free();
    if (ht_init(256) != 0) {
        printf("  [FAIL] 8.5 ht_init(256) failed\n");
        fail_count++;
        return;
    }

    int n = 100; /* insert 100 entries, load factor 100/256 ≈ 0.39 */
    uint8_t keys[100][20];
    for (int i = 0; i < n; i++) {
        memset(keys[i], 0, 20);
        keys[i][0] = (uint8_t)(i >> 8);
        keys[i][1] = (uint8_t)(i & 0xff);
        keys[i][19] = (uint8_t)(i * 7 + 3); /* increase diversity */
        ht_insert(keys[i]);
    }

    int all_found = 1;
    for (int i = 0; i < n; i++) {
        if (!ht_contains(keys[i])) {
            printf("  [FAIL] 8.5 bulk insert: entry %d missed\n", i);
            all_found = 0;
            fail_count++;
            break;
        }
    }
    if (all_found) {
        printf("  [PASS] 8.5 bulk insert (100 entries) all hit\n");
        pass_count++;
    }

    /* 8.6 Verify keys not inserted do not false-hit */
    uint8_t h_miss[20] = {0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,
                          0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,0xef,0xee,0xed,0xec};
    if (ht_contains(h_miss) == 0) {
        printf("  [PASS] 8.6 uninserted key does not false-hit\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.6 uninserted key false-hit\n");
        fail_count++;
    }

    ht_free();
}

#ifdef __AVX2__
static void test_ht_contains_8way_func(void) {
    printf("\n=== ht_contains_8way AVX2 batch lookup tests ===\n");

    /* Initialize hash table */
    if (ht_init(64) != 0) {
        printf("  [FAIL] 9.0 ht_init failed\n");
        fail_count++;
        return;
    }

    /* Prepare 8 known hash160 values */
    uint8_t known[8][20];
    for (int i = 0; i < 8; i++) {
        memset(known[i], 0, 20);
        known[i][0] = (uint8_t)(0x10 + i);
        known[i][1] = (uint8_t)(0x20 + i);
        known[i][2] = (uint8_t)(0x30 + i);
        known[i][3] = (uint8_t)(0x40 + i);
        known[i][19] = (uint8_t)(i + 1);
        ht_insert(known[i]);
    }

    /* Prepare 8 hash160 values not inserted */
    uint8_t unknown[8][20];
    for (int i = 0; i < 8; i++) {
        memset(unknown[i], 0, 20);
        unknown[i][0] = (uint8_t)(0xf0 + i);
        unknown[i][1] = (uint8_t)(0xe0 + i);
        unknown[i][2] = (uint8_t)(0xd0 + i);
        unknown[i][3] = (uint8_t)(0xc0 + i);
        unknown[i][19] = (uint8_t)(i + 0x80);
    }

    /* 9.1 all 8 lanes hit */
    const uint8_t *ptrs_all[8];
    for (int i = 0; i < 8; i++) ptrs_all[i] = known[i];
    uint8_t mask = ht_contains_8way(ptrs_all);
    if (mask == 0xff) {
        printf("  [PASS] 9.1 all 8 lanes hit, mask=0xff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.1 all 8 lanes hit, expected mask=0xff, actual=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.2 all 8 lanes miss */
    const uint8_t *ptrs_none[8];
    for (int i = 0; i < 8; i++) ptrs_none[i] = unknown[i];
    mask = ht_contains_8way(ptrs_none);
    if (mask == 0x00) {
        printf("  [PASS] 9.2 all 8 lanes miss, mask=0x00\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.2 all 8 lanes miss, expected mask=0x00, actual=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.3 partial hit: even lanes hit (0,2,4,6), expected mask=0x55 */
    const uint8_t *ptrs_mix[8];
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i % 2 == 0) ? known[i] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x55) {
        printf("  [PASS] 9.3 partial hit (even lanes), mask=0x55\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.3 partial hit (even lanes), expected mask=0x55, actual=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.4 partial hit: odd lanes hit (1,3,5,7), expected mask=0xaa */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i % 2 == 1) ? known[i] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0xaa) {
        printf("  [PASS] 9.4 partial hit (odd lanes), mask=0xaa\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.4 partial hit (odd lanes), expected mask=0xaa, actual=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.5 only lane0 hits, expected mask=0x01 */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i == 0) ? known[0] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x01) {
        printf("  [PASS] 9.5 only lane0 hits, mask=0x01\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.5 only lane0 hits, expected mask=0x01, actual=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.6 only lane7 hits, expected mask=0x80 */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i == 7) ? known[7] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x80) {
        printf("  [PASS] 9.6 only lane7 hits, mask=0x80\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.6 only lane7 hits, expected mask=0x80, actual=0x%02x\n", mask);
        fail_count++;
    }

    ht_free();
}
#endif /* __AVX2__ */

/* ===================== Specialized Interface Tests: sha256_33 / sha256_65 / ripemd160_32 ===================== */

static void test_specialized_interfaces(void) {
    printf("\n=== Specialized Interface Tests: sha256_33 / sha256_65 / ripemd160_32 ===\n");

    uint8_t digest_spec[32];
    uint8_t digest_ref[32];
    uint8_t rmd_spec[20];
    uint8_t rmd_ref[20];
    char ref_hex[65];

    /* ------------------------------------------------------------------ */
    /* 6.1  sha256_33 — G point compressed pubkey known vector
     *   input: 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   expected: 0b7c28c9b7290c98d7438e70b3d3f7c848fbd7d1dc194ff83f4f7cc9b1378e98
     *   (hash160 = ripemd160(sha256(G)) = 751e76e8199196d454941c45d1b3a323f1433bd6 known)
     * ------------------------------------------------------------------ */
    {
        static const uint8_t G_comp[33] = {
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
            0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
            0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
            0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
        };
        /* Compute reference value using generic interface */
        sha256(G_comp, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        /* Compute using specialized interface */
        sha256_33(G_comp, digest_spec);
        check("6.1 sha256_33(G point compressed pubkey) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.2  sha256_33 — all-zero 33 bytes (self-consistency) */
    {
        uint8_t zeros33[33];
        memset(zeros33, 0x00, 33);
        sha256(zeros33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(zeros33, digest_spec);
        check("6.2 sha256_33(all-zero 33 bytes) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.3  sha256_33 — all-0xFF 33 bytes (self-consistency) */
    {
        uint8_t ff33[33];
        memset(ff33, 0xFF, 33);
        sha256(ff33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(ff33, digest_spec);
        check("6.3 sha256_33(all-0xFF 33 bytes) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.4  sha256_33 — first byte 0x02, rest incrementing (self-consistency) */
    {
        uint8_t incr33[33];
        incr33[0] = 0x02;
        for (int i = 1; i < 33; i++) incr33[i] = (uint8_t)(i);
        sha256(incr33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(incr33, digest_spec);
        check("6.4 sha256_33(0x02+incrementing sequence) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.5  sha256_65 — G point uncompressed pubkey known vector (self-consistency) */
    {
        static const uint8_t G_uncomp[65] = {
            0x04,
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
            0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
            0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
            0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
            0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
            0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
            0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
            0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
        };
        sha256(G_uncomp, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(G_uncomp, digest_spec);
        check("6.5 sha256_65(G point uncompressed pubkey) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.6  sha256_65 — all-zero 65 bytes (self-consistency) */
    {
        uint8_t zeros65[65];
        memset(zeros65, 0x00, 65);
        sha256(zeros65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(zeros65, digest_spec);
        check("6.6 sha256_65(all-zero 65 bytes) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.7  sha256_65 — all-0xFF 65 bytes (self-consistency) */
    {
        uint8_t ff65[65];
        memset(ff65, 0xFF, 65);
        sha256(ff65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(ff65, digest_spec);
        check("6.7 sha256_65(all-0xFF 65 bytes) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.8  sha256_65 — first byte 0x04, rest incrementing (self-consistency) */
    {
        uint8_t incr65[65];
        incr65[0] = 0x04;
        for (int i = 1; i < 65; i++) incr65[i] = (uint8_t)(i);
        sha256(incr65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(incr65, digest_spec);
        check("6.8 sha256_65(0x04+incrementing sequence) consistent with generic sha256", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.9  ripemd160_32 — SHA256 result of G point compressed pubkey (known vector)
     *   input: sha256(G_compressed) = 0b7c28c9b7290c98d7438e70b3d3f7c848fbd7d1dc194ff83f4f7cc9b1378e98
     *   expected: hash160(G) = 751e76e8199196d454941c45d1b3a323f1433bd6
     * ------------------------------------------------------------------ */
    {
        static const uint8_t G_comp[33] = {
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
            0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
            0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
            0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
        };
        /* First compute SHA256 of G using generic sha256 (32 bytes) */
        uint8_t sha256_G[32];
        sha256(G_comp, 33, sha256_G);
        /* Compute using specialized interfaceripemd160_32 */
        ripemd160_32(sha256_G, rmd_spec);
        check("6.9 ripemd160_32(sha256(G)) consistent with known hash160 vector",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.10 ripemd160_32 — all-zero 32 bytes (self-consistency) */
    {
        uint8_t zeros32[32];
        memset(zeros32, 0x00, 32);
        ripemd160(zeros32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(zeros32, rmd_spec);
        check("6.10 ripemd160_32(all-zero 32 bytes) consistent with generic ripemd160", rmd_ref_hex, rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.11 ripemd160_32 — all-0xFF 32 bytes (self-consistency) */
    {
        uint8_t ff32[32];
        memset(ff32, 0xFF, 32);
        ripemd160(ff32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(ff32, rmd_spec);
        check("6.11 ripemd160_32(all-0xFF 32 bytes) consistent with generic ripemd160", rmd_ref_hex, rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.12 ripemd160_32 — incrementing sequence 0x00~0x1F (self-consistency) */
    {
        uint8_t incr32[32];
        for (int i = 0; i < 32; i++) incr32[i] = (uint8_t)i;
        ripemd160(incr32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(incr32, rmd_spec);
        check("6.12 ripemd160_32(0x00~0x1F incrementing sequence) consistent with generic ripemd160", rmd_ref_hex, rmd_spec, 20);
    }
}

/* ===================== Bech32 Encode/Decode Tests ===================== */

static void test_bech32(void) {
    printf("\n=== Bech32 Encode/Decode Tests ===\n");

    int witness_ver;
    uint8_t witness_prog[40];
    size_t witness_len;
    char encoded[90];
    int ret;

    /* ------------------------------------------------------------------ */
    /* 7.1 Decode a known P2WPKH address (witness v0, 20-byte program)    */
    /* Address: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4                */
    /* Expected witness program: 751e76e8199196d454941c45d1b3a323f1433bd6  */
    /* ------------------------------------------------------------------ */
    {
        const char *addr = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";
        ret = bech32_decode_witness(addr, &witness_ver, witness_prog, &witness_len);
        if (ret != 0) {
            printf("  [FAIL] 7.1 bech32_decode P2WPKH: decode returned %d\n", ret);
            fail_count++;
        } else if (witness_ver != 0 || witness_len != 20) {
            printf("  [FAIL] 7.1 bech32_decode P2WPKH: witness_ver=%d, witness_len=%zu (expected 0, 20)\n",
                   witness_ver, witness_len);
            fail_count++;
        } else {
            check("7.1 bech32_decode P2WPKH witness program",
                  "751e76e8199196d454941c45d1b3a323f1433bd6",
                  witness_prog, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.2 Encode the same witness program back and verify round-trip     */
    /* ------------------------------------------------------------------ */
    {
        const uint8_t prog[20] = {
            0x75, 0x1e, 0x76, 0xe8, 0x19, 0x91, 0x96, 0xd4, 0x54, 0x94,
            0x1c, 0x45, 0xd1, 0xb3, 0xa3, 0x23, 0xf1, 0x43, 0x3b, 0xd6
        };
        ret = bech32_encode_witness("bc", 0, prog, 20, encoded);
        if (ret != 0) {
            printf("  [FAIL] 7.2 bech32_encode P2WPKH: encode returned %d\n", ret);
            fail_count++;
        } else if (strcmp(encoded, "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4") == 0) {
            printf("  [PASS] 7.2 bech32_encode P2WPKH round-trip\n");
            pass_count++;
        } else {
            printf("  [FAIL] 7.2 bech32_encode P2WPKH round-trip\n");
            printf("         expected: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4\n");
            printf("         actual:   %s\n", encoded);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.3 Decode a known P2WSH address (witness v0, 32-byte program)     */
    /* Address: bc1qwqdg6squsna38e46795at95yu9atm8azzmyvckulcc7kytlcckxswvvzej */
    /* Expected witness program:                                          */
    /*   701a8d401c84fb13e6baf169d59684e17abd9fa216c8cc5b9fc63d622ff8c58d  */
    /* ------------------------------------------------------------------ */
    {
        const char *addr = "bc1qwqdg6squsna38e46795at95yu9atm8azzmyvckulcc7kytlcckxswvvzej";
        ret = bech32_decode_witness(addr, &witness_ver, witness_prog, &witness_len);
        if (ret != 0) {
            printf("  [FAIL] 7.3 bech32_decode P2WSH: decode returned %d\n", ret);
            fail_count++;
        } else if (witness_ver != 0 || witness_len != 32) {
            printf("  [FAIL] 7.3 bech32_decode P2WSH: witness_ver=%d, witness_len=%zu (expected 0, 32)\n",
                   witness_ver, witness_len);
            fail_count++;
        } else {
            check("7.3 bech32_decode P2WSH witness program (32 bytes)",
                  "701a8d401c84fb13e6baf169d59684e17abd9fa216c8cc5b9fc63d622ff8c58d",
                  witness_prog, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.4 Decode + re-encode round-trip for another P2WPKH address       */
    /* Address: bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq                */
    /* Expected witness program: e8df018c7e28e1b9b1a763a89f5971ec7963153c  */
    /* (This is a well-known Bitcoin address)                              */
    /* ------------------------------------------------------------------ */
    {
        const char *addr = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq";
        ret = bech32_decode_witness(addr, &witness_ver, witness_prog, &witness_len);
        if (ret != 0) {
            printf("  [FAIL] 7.4 bech32_decode P2WPKH #2: decode returned %d\n", ret);
            fail_count++;
        } else if (witness_ver != 0 || witness_len != 20) {
            printf("  [FAIL] 7.4 bech32_decode P2WPKH #2: witness_ver=%d, witness_len=%zu\n",
                   witness_ver, witness_len);
            fail_count++;
        } else {
            check("7.4 bech32_decode P2WPKH #2 witness program",
                  "e8df018c7e326cc253faac7e46cdc51e68542c42",
                  witness_prog, 20);

            /* Re-encode and verify */
            ret = bech32_encode_witness("bc", 0, witness_prog, 20, encoded);
            if (ret != 0) {
                printf("  [FAIL] 7.4 bech32_encode P2WPKH #2: encode returned %d\n", ret);
                fail_count++;
            } else if (strcmp(encoded, addr) == 0) {
                printf("  [PASS] 7.4 bech32_encode P2WPKH #2 round-trip\n");
                pass_count++;
            } else {
                printf("  [FAIL] 7.4 bech32_encode P2WPKH #2 round-trip\n");
                printf("         expected: %s\n", addr);
                printf("         actual:   %s\n", encoded);
                fail_count++;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.5 Encode from raw hash160 bytes and decode back (full round-trip)*/
    /* Use all-zero 20-byte witness program                               */
    /* ------------------------------------------------------------------ */
    {
        const uint8_t zero_prog[20] = {0};
        ret = bech32_encode_witness("bc", 0, zero_prog, 20, encoded);
        if (ret != 0) {
            printf("  [FAIL] 7.5 bech32_encode all-zero P2WPKH: encode returned %d\n", ret);
            fail_count++;
        } else {
            /* Decode back */
            ret = bech32_decode_witness(encoded, &witness_ver, witness_prog, &witness_len);
            if (ret != 0) {
                printf("  [FAIL] 7.5 bech32 round-trip all-zero: decode returned %d\n", ret);
                fail_count++;
            } else if (witness_ver != 0 || witness_len != 20) {
                printf("  [FAIL] 7.5 bech32 round-trip all-zero: witness_ver=%d, witness_len=%zu\n",
                       witness_ver, witness_len);
                fail_count++;
            } else {
                check("7.5 bech32 round-trip all-zero P2WPKH",
                      "0000000000000000000000000000000000000000",
                      witness_prog, 20);
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.6 Encode from all-0xFF 20-byte witness program and decode back   */
    /* ------------------------------------------------------------------ */
    {
        const uint8_t ff_prog[20] = {
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
        };
        ret = bech32_encode_witness("bc", 0, ff_prog, 20, encoded);
        if (ret != 0) {
            printf("  [FAIL] 7.6 bech32_encode all-0xFF P2WPKH: encode returned %d\n", ret);
            fail_count++;
        } else {
            ret = bech32_decode_witness(encoded, &witness_ver, witness_prog, &witness_len);
            if (ret != 0) {
                printf("  [FAIL] 7.6 bech32 round-trip all-0xFF: decode returned %d\n", ret);
                fail_count++;
            } else {
                check("7.6 bech32 round-trip all-0xFF P2WPKH",
                      "ffffffffffffffffffffffffffffffffffffffff",
                      witness_prog, 20);
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.7 Error case: invalid checksum should return -2                  */
    /* ------------------------------------------------------------------ */
    {
        /* Corrupt the last character of a valid address */
        const char *bad_addr = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5";
        ret = bech32_decode_witness(bad_addr, &witness_ver, witness_prog, &witness_len);
        if (ret == -2) {
            printf("  [PASS] 7.7 bech32_decode invalid checksum correctly rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.7 bech32_decode invalid checksum: expected ret=-2, got ret=%d\n", ret);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.8 Error case: non-bc HRP should return -1                        */
    /* ------------------------------------------------------------------ */
    {
        const char *tb_addr = "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx";
        ret = bech32_decode_witness(tb_addr, &witness_ver, witness_prog, &witness_len);
        if (ret == -1) {
            printf("  [PASS] 7.8 bech32_decode non-bc HRP correctly rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.8 bech32_decode non-bc HRP: expected ret=-1, got ret=%d\n", ret);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.9 Error case: empty string should return -1                      */
    /* ------------------------------------------------------------------ */
    {
        ret = bech32_decode_witness("", &witness_ver, witness_prog, &witness_len);
        if (ret == -1) {
            printf("  [PASS] 7.9 bech32_decode empty string correctly rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.9 bech32_decode empty string: expected ret=-1, got ret=%d\n", ret);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.10 Error case: mixed case should return -1                       */
    /* ------------------------------------------------------------------ */
    {
        const char *mixed = "bc1qW508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";
        ret = bech32_decode_witness(mixed, &witness_ver, witness_prog, &witness_len);
        if (ret == -1 || ret == -2) {
            printf("  [PASS] 7.10 bech32_decode mixed case correctly rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.10 bech32_decode mixed case: expected ret=-1 or -2, got ret=%d\n", ret);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.11 Encode with invalid parameters should return -1               */
    /* ------------------------------------------------------------------ */
    {
        const uint8_t prog[20] = {0};
        /* witness_ver out of range */
        ret = bech32_encode_witness("bc", 17, prog, 20, encoded);
        if (ret == -1) {
            printf("  [PASS] 7.11 bech32_encode invalid witness_ver=17 rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.11 bech32_encode invalid witness_ver=17: expected ret=-1, got ret=%d\n", ret);
            fail_count++;
        }

        /* witness_len too small */
        ret = bech32_encode_witness("bc", 0, prog, 1, encoded);
        if (ret == -1) {
            printf("  [PASS] 7.11 bech32_encode witness_len=1 rejected (ret=%d)\n", ret);
            pass_count++;
        } else {
            printf("  [FAIL] 7.11 bech32_encode witness_len=1: expected ret=-1, got ret=%d\n", ret);
            fail_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.12 Decode uppercase bech32 address (BIP173 allows all-uppercase) */
    /* ------------------------------------------------------------------ */
    {
        const char *upper_addr = "BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4";
        ret = bech32_decode_witness(upper_addr, &witness_ver, witness_prog, &witness_len);
        if (ret != 0) {
            printf("  [FAIL] 7.12 bech32_decode uppercase: decode returned %d\n", ret);
            fail_count++;
        } else {
            check("7.12 bech32_decode uppercase P2WPKH",
                  "751e76e8199196d454941c45d1b3a323f1433bd6",
                  witness_prog, 20);
        }
    }
}

/* ===================== keygen Internal Interface Correctness Tests ===================== */

#ifndef USE_PUBKEY_API_ONLY

static void test_keygen_internal(void) {
    printf("\n=== keygen Internal Interface Correctness Tests ===\n");

    /* ------------------------------------------------------------------
     * 5.1  keygen_init_generator: verify G_affine.infinity == 0
     *      and G_affine.x / G_affine.y match known standard values
     * ------------------------------------------------------------------ */
    {
        /* Known compressed pubkey bytes of G (pubkey corresponding to privkey k=1 is G) */
        static const uint8_t G_compressed_expected[33] = {
            0x02,
            0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
            0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
            0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
            0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
        };

        /* Extract compressed bytes from G_affine using keygen_ge_to_pubkey_bytes */
        uint8_t G_bytes[33];
        keygen_ge_to_pubkey_bytes(&G_affine, G_bytes, NULL);

        char expected_hex[67];
        bytes_to_hex_helper(G_compressed_expected, 33, expected_hex);
        check("5.1 keygen_init_generator: G compressed pubkey matches known standard value",
              expected_hex, G_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.2  keygen_privkey_to_gej + keygen_batch_normalize(n=1)：
     *      verify k=1 Jacobian point after normalization matches direct serialize result
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        /* Method A: keygen_privkey_to_gej + keygen_batch_normalize + keygen_ge_to_pubkey_bytes */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t keygen_bytes[33];
        if (keygen_privkey_to_gej(secp_ctx, privkey1, &gej) == 0) {
            keygen_batch_normalize(&gej, &ge, 1);
            keygen_ge_to_pubkey_bytes(&ge, keygen_bytes, NULL);
        } else {
            memset(keygen_bytes, 0, 33);
        }

        /* Method B: public API serialize */
        uint8_t api_bytes[33];
        privkey_to_compressed_bytes(privkey1, api_bytes);

        char api_hex[67];
        bytes_to_hex_helper(api_bytes, 33, api_hex);
        check("5.2 keygen_privkey_to_gej + batch_normalize(n=1) matches public API (k=1)",
              api_hex, keygen_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.3  Direct point addition secp256k1_gej_add_ge + keygen_batch_normalize:
     *      verify k=1 Jacobian point after adding G and normalizing matches direct computation of k=2
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;

        /* Method A: gej(k=1) + G_affine -> normalize -> bytes */
        secp256k1_gej gej_k1, gej_k2;
        secp256k1_ge  ge_k2;
        uint8_t incr_bytes[33];
        keygen_privkey_to_gej(secp_ctx, privkey1, &gej_k1);
        secp256k1_gej_add_ge(&gej_k2, &gej_k1, &G_affine);
        keygen_batch_normalize(&gej_k2, &ge_k2, 1);
        keygen_ge_to_pubkey_bytes(&ge_k2, incr_bytes, NULL);

        /* Method B: compute k=2 directly */
        uint8_t direct_bytes[33];
        privkey_to_compressed_bytes(privkey2, direct_bytes);

        char direct_hex[67];
        bytes_to_hex_helper(direct_bytes, 33, direct_hex);
        check("5.3 gej_add_ge(k=1, G) + batch_normalize matches direct computation of k=2",
              direct_hex, incr_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.4  keygen_batch_normalize(n=BATCH_SIZE)：
     *      batch normalize 2048 points, verify each point matches individual serialize result
     * ------------------------------------------------------------------ */
    {
        printf("  [batch normalize BATCH_SIZE=2048 verification]\n");

        const int N = 2048;
        secp256k1_gej *gej_arr = (secp256k1_gej *)malloc(N * sizeof(secp256k1_gej));
        secp256k1_ge  *ge_arr  = (secp256k1_ge  *)malloc(N * sizeof(secp256k1_ge));

        if (!gej_arr || !ge_arr) {
            printf("  [SKIP] 5.4 memory allocation failed, skipping\n");
            free(gej_arr); free(ge_arr);
        } else {
            /* Starting from k=1, continuously add G to generate 2048 Jacobian points */
            uint8_t privkey[32] = {0};
            privkey[31] = 1;
            secp256k1_gej cur;
            keygen_privkey_to_gej(secp_ctx, privkey, &cur);
            for (int i = 0; i < N; i++) {
                gej_arr[i] = cur;
                secp256k1_gej next;
                secp256k1_gej_add_ge(&next, &cur, &G_affine);
                cur = next;
            }

            /* Batch normalize */
            keygen_batch_normalize(gej_arr, ge_arr, (size_t)N);

            /* Verify each point: keygen_ge_to_pubkey_bytes matches public API serialize */
            int all_pass = 1;
            uint8_t privkey_i[32] = {0};
            privkey_i[31] = 1;
            for (int i = 0; i < N; i++) {
                uint8_t keygen_bytes[33];
                uint8_t api_bytes[33];

                keygen_ge_to_pubkey_bytes(&ge_arr[i], keygen_bytes, NULL);
                privkey_to_compressed_bytes(privkey_i, api_bytes);

                if (memcmp(keygen_bytes, api_bytes, 33) != 0) {
                    char kb[67], ab[67];
                    bytes_to_hex_helper(keygen_bytes, 33, kb);
                    bytes_to_hex_helper(api_bytes,    33, ab);
                    printf("  [FAIL] 5.4 point %d mismatch:\n"
                           "         keygen: %s\n"
                           "         api:    %s\n", i, kb, ab);
                    all_pass = 0;
                    fail_count++;
                    break;
                }

                /* Increment private key, corresponding to next point */
                uint8_t tweak[32] = {0};
                tweak[31] = 1;
                secp256k1_ec_seckey_tweak_add(secp_ctx, privkey_i, tweak);
            }
            if (all_pass) {
                printf("  [PASS] 5.4 batch normalize 2048 points all match public API\n");
                pass_count++;
            }

            free(gej_arr);
            free(ge_arr);
        }
    }

    /* ------------------------------------------------------------------
     * 5.5  keygen_ge_to_pubkey_bytes uncompressed format verification:
     *      verify k=1 uncompressed pubkey bytes match public API serialize
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        /* Method A: keygen path */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t keygen_uncomp[65];
        keygen_privkey_to_gej(secp_ctx, privkey1, &gej);
        keygen_batch_normalize(&gej, &ge, 1);
        keygen_ge_to_pubkey_bytes(&ge, NULL, keygen_uncomp);

        /* Method B: public API */
        secp256k1_pubkey pubkey;
        uint8_t api_uncomp[65];
        size_t len = 65;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey1);
        secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len,
                                      &pubkey, SECP256K1_EC_UNCOMPRESSED);

        char api_hex[131];
        bytes_to_hex_helper(api_uncomp, 65, api_hex);
        check("5.5 keygen_ge_to_pubkey_bytes uncompressed format matches public API (k=1)",
              api_hex, keygen_uncomp, 65);
    }

    /* ------------------------------------------------------------------
     * 5.6  keygen path hash160 vs privkey_to_hash160 consistency:
     *      verify hash160 obtained via keygen path for k=5 matches direct computation
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey5[32] = {0};
        privkey5[31] = 5;

        /* Method A: keygen path */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t comp_bytes[33], uncomp_bytes[65];
        uint8_t hash160_keygen_comp[20], hash160_keygen_uncomp[20];
        keygen_privkey_to_gej(secp_ctx, privkey5, &gej);
        keygen_batch_normalize(&gej, &ge, 1);
        keygen_ge_to_pubkey_bytes(&ge, comp_bytes, uncomp_bytes);
        pubkey_bytes_to_hash160(comp_bytes,   33, hash160_keygen_comp);
        pubkey_bytes_to_hash160(uncomp_bytes, 65, hash160_keygen_uncomp);

        /* Method B: privkey_to_hash160 */
        uint8_t hash160_direct_comp[20], hash160_direct_uncomp[20];
        privkey_to_hash160(privkey5, hash160_direct_comp, hash160_direct_uncomp);

        char direct_comp_hex[41], direct_uncomp_hex[41];
        bytes_to_hex_helper(hash160_direct_comp,   20, direct_comp_hex);
        bytes_to_hex_helper(hash160_direct_uncomp, 20, direct_uncomp_hex);

        check("5.6a keygen path compressed pubkey hash160 matches privkey_to_hash160 (k=5)",
              direct_comp_hex, hash160_keygen_comp, 20);
        check("5.6b keygen path uncompressed pubkey hash160 matches privkey_to_hash160 (k=5)",
              direct_uncomp_hex, hash160_keygen_uncomp, 20);
    }

    /* ------------------------------------------------------------------
     * 5.7  Multi-step point addition + hash160 consistency after batch normalize (10 steps):
     *      continuously add G from k=3 for 10 steps, verify each step's hash160 matches direct computation
     * ------------------------------------------------------------------ */
    {
        printf("  [multi-step point addition + batch normalize hash160 verification, k=3..13]\n");

        const int STEPS = 10;
        secp256k1_gej gej_arr[10];
        secp256k1_ge  ge_arr[10];

        uint8_t privkey[32] = {0};
        privkey[31] = 3;
        secp256k1_gej cur;
        keygen_privkey_to_gej(secp_ctx, privkey, &cur);

        uint8_t tweak[32] = {0};
        tweak[31] = 1;

        /* Accumulate 10 Jacobian points (k=4..13, i.e. results after adding G) */
        for (int i = 0; i < STEPS; i++) {
            secp256k1_gej next;
            secp256k1_gej_add_ge(&next, &cur, &G_affine);
            cur = next;
            gej_arr[i] = cur;
        }

        /* Batch normalize */
        keygen_batch_normalize(gej_arr, ge_arr, (size_t)STEPS);

        /* Verify hash160 step by step */
        int all_pass = 1;
        uint8_t privkey_i[32] = {0};
        privkey_i[31] = 4;  /* first step corresponds to k=4 */
        for (int i = 0; i < STEPS; i++) {
            uint8_t comp_bytes[33];
            uint8_t hash160_keygen[20], hash160_direct[20];

            keygen_ge_to_pubkey_bytes(&ge_arr[i], comp_bytes, NULL);
            pubkey_bytes_to_hash160(comp_bytes, 33, hash160_keygen);
            privkey_to_hash160(privkey_i, hash160_direct, NULL);

            if (memcmp(hash160_keygen, hash160_direct, 20) != 0) {
                char kh[41], dh[41];
                bytes_to_hex_helper(hash160_keygen, 20, kh);
                bytes_to_hex_helper(hash160_direct, 20, dh);
                printf("  [FAIL] 5.7 step %d (k=%d): keygen=%s direct=%s\n",
                       i, i + 4, kh, dh);
                all_pass = 0;
                fail_count++;
            }
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey_i, tweak);
        }
        if (all_pass) {
            printf("  [PASS] 5.7 multi-step point addition + batch normalize hash160 (all 10 steps consistent)\n");
            pass_count++;
        }
    }
}

/* ------------------------------------------------------------------
 * test_search_key_privkey_pubkey
 *
 * Specialized test: simulate private key iteration and pubkey derivation in search_key,
 * verify internal path (scalar accumulation + gej_add_ge_var + batch_normalize_rzr)
 * is fully consistent with standard API (secp256k1_ec_pubkey_create + serialize).
 *
 * Coverage scenarios:
 *   6.1  Single batch head/tail (b=0 and b=BATCH_SIZE-1) privkey->pubkey consistency
 *   6.2  rzr path batch_normalize_rzr consistent with batch_normalize results
 *   6.3  Hit reconstruction logic: pubkey for hit_scalar = base + b_idx * tweak
 *        consistent with ge_batch[b_idx] (verify b_idx=0/middle/end three positions)
 *   6.4  Full batch (BATCH_SIZE=4096) each point pubkey consistent with standard API
 *   6.5  scalar overflow boundary: base near curve order n, verify overflow detection
 *   6.6  Uncompressed pubkey path: ge_batch uncompressed bytes consistent with standard API
 * ------------------------------------------------------------------ */
static void test_search_key_privkey_pubkey(void) {
    printf("\n=== search_key private key iteration and pubkey derivation consistency tests ===\n");

    /* tweak_scalar = 1 (consistent with search_key) */
    secp256k1_scalar tweak_scalar;
    secp256k1_scalar_set_int(&tweak_scalar, 1);

    /* tweak bytes for public API (value = 1) */
    uint8_t tweak_bytes[32] = {0};
    tweak_bytes[31] = 1;

    /* ------------------------------------------------------------------ */
    /* 6.1  Single batch head/tail (b=0 and b=BATCH_SIZE-1) privkey->pubkey consistency
     *
     *   Simulate search_key outer loop:
     *     base_privkey_scalar = k (random base, using k=7 here for easy verification)
     *     cur_privkey_scalar  = base_privkey_scalar
     *     gej_batch[0]        = keygen_privkey_to_gej(base)
     *     gej_batch[b]        = gej_batch[b-1] + G
     *
     *   Verify:
     *     ge_batch[0]  pubkey for privkey base+0
     *     ge_batch[N-1] pubkey for privkey base+(N-1)
     * ------------------------------------------------------------------ */
    {
        const int N = 16;  /* small batch, quick head/tail verification */
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 7;  /* base = 7 */

        secp256k1_scalar base_scalar, cur_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);

        /* Build gej_batch and rzr_batch (simulate search_key inner loop) */
        secp256k1_gej gej_batch[16];
        secp256k1_ge  ge_batch[16];
        secp256k1_fe  rzr_batch[16];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);
        cur_scalar = base_scalar;

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak_scalar);
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }

        /* Batch normalize (rzr path) */
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        /* Verify b=0: ge_batch[0] corresponds to privkey base+0=7 */
        {
            uint8_t keygen_comp[33], api_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[0], keygen_comp, NULL);

            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, base_privkey);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            char api_hex[67];
            bytes_to_hex_helper(api_comp, 33, api_hex);
            check("6.1a batch head b=0: ge_batch[0] consistent with standard API (base=7)",
                  api_hex, keygen_comp, 33);
        }

        /* Verify b=N-1: ge_batch[N-1] corresponds to privkey base+(N-1)=7+15=22 */
        {
            uint8_t privkey_end[32] = {0};
            privkey_end[31] = 7 + (N - 1);  /* = 22 */

            uint8_t keygen_comp[33], api_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[N - 1], keygen_comp, NULL);

            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey_end);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            char api_hex[67];
            bytes_to_hex_helper(api_comp, 33, api_hex);
            check("6.1b batch tail b=N-1: ge_batch[N-1] consistent with standard API (base+15=22)",
                  api_hex, keygen_comp, 33);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.2  rzr path batch_normalize_rzr consistent with batch_normalize results
     *
     *   For the same gej_batch, use two normalization methods,
     *   verify compressed pubkey bytes of each point are identical.
     * ------------------------------------------------------------------ */
    {
        printf("  [rzr path vs standard path batch_normalize consistency, N=32]\n");
        const int N = 32;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 100;  /* base = 100 */

        secp256k1_gej gej_batch[32];
        secp256k1_ge  ge_rzr[32], ge_std[32];
        secp256k1_fe  rzr_batch[32];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }

        /* Two normalization methods */
        keygen_batch_normalize_rzr(gej_batch, ge_rzr, rzr_batch, (size_t)N);
        keygen_batch_normalize(gej_batch, ge_std, (size_t)N);

        int all_pass = 1;
        for (int b = 0; b < N; b++) {
            uint8_t comp_rzr[33], comp_std[33];
            keygen_ge_to_pubkey_bytes(&ge_rzr[b], comp_rzr, NULL);
            keygen_ge_to_pubkey_bytes(&ge_std[b], comp_std, NULL);
            if (memcmp(comp_rzr, comp_std, 33) != 0) {
                char rzr_hex[67], std_hex[67];
                bytes_to_hex_helper(comp_rzr, 33, rzr_hex);
                bytes_to_hex_helper(comp_std, 33, std_hex);
                printf("  [FAIL] 6.2 b=%d: rzr=%s std=%s\n", b, rzr_hex, std_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.2 rzr path and standard path batch_normalize results fully consistent (N=32)\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.3  Hit reconstruction logic verification
     *
     *   Simulate privkey reconstruction on search_key hit:
     *     hit_scalar = base_privkey_scalar
     *     for i in range(b_idx): hit_scalar += tweak_scalar
     *     hit_privkey = scalar_get_b32(hit_scalar)
     *
     *   Verify three positions (b_idx=0, middle, end):
     *     pubkey of reconstructed privkey == pubkey bytes of ge_batch[b_idx]
     * ------------------------------------------------------------------ */
    {
        const int N = 64;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 50;  /* base = 50 */

        secp256k1_scalar base_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);

        secp256k1_gej gej_batch[64];
        secp256k1_ge  ge_batch[64];
        secp256k1_fe  rzr_batch[64];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        /* Test three hit positions */
        int test_positions[] = {0, N / 2, N - 1};
        const char *pos_names[] = {"b_idx=0 (head)", "b_idx=N/2 (middle)", "b_idx=N-1 (tail)"};
        int all_pass = 1;

        for (int t = 0; t < 3; t++) {
            int b_idx = test_positions[t];

            /* Simulate hit reconstruction: hit_scalar = base + b_idx * tweak */
            secp256k1_scalar hit_scalar = base_scalar;
            for (int i = 0; i < b_idx; i++) {
                secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
            }
            uint8_t hit_privkey[32];
            secp256k1_scalar_get_b32(hit_privkey, &hit_scalar);

            /* Pubkey of reconstructed privkey (standard API) */
            uint8_t api_comp[33];
            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, hit_privkey);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);

            /* Pubkey bytes of ge_batch[b_idx] */
            uint8_t batch_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[b_idx], batch_comp, NULL);

            if (memcmp(api_comp, batch_comp, 33) != 0) {
                char api_hex[67], batch_hex[67];
                bytes_to_hex_helper(api_comp,   33, api_hex);
                bytes_to_hex_helper(batch_comp, 33, batch_hex);
                printf("  [FAIL] 6.3 hit reconstruction %s:\n"
                       "         reconstructed privkey pubkey: %s\n"
                       "         ge_batch pubkey: %s\n",
                       pos_names[t], api_hex, batch_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.3 hit reconstruction logic: three positions (head/middle/tail) privkey pubkey consistent with ge_batch\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.4  Full batch (BATCH_SIZE=4096) each point pubkey consistent with standard API
     *
     *   Simulate full search_key inner loop, verify pubkey correctness of all 4096 points.
     *   Sampling strategy: verify first, last, and every 256th point (18 points total).
     * ------------------------------------------------------------------ */
    {
        printf("  [Full batch BATCH_SIZE=4096 sampling verification]\n");
        const int N = 4096;

        secp256k1_gej *gej_batch = (secp256k1_gej *)malloc(N * sizeof(secp256k1_gej));
        secp256k1_ge  *ge_batch  = (secp256k1_ge  *)malloc(N * sizeof(secp256k1_ge));
        secp256k1_fe  *rzr_batch = (secp256k1_fe  *)malloc(N * sizeof(secp256k1_fe));

        if (!gej_batch || !ge_batch || !rzr_batch) {
            printf("  [SKIP] 6.4 memory allocation failed, skipping\n");
            free(gej_batch); free(ge_batch); free(rzr_batch);
        } else {
            /* base = 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20 */
            uint8_t base_privkey[32];
            for (int i = 0; i < 32; i++) base_privkey[i] = (uint8_t)(i + 1);

            secp256k1_scalar base_scalar, cur_scalar;
            int overflow = 0;
            secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);
            cur_scalar = base_scalar;

            secp256k1_gej cur_gej;
            keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

            /* Build full batch (simulate search_key inner loop) */
            for (int b = 0; b < N; b++) {
                gej_batch[b] = cur_gej;
                if (b < N - 1) {
                    secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak_scalar);
                    secp256k1_gej next_gej;
                    secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                    cur_gej = next_gej;
                }
            }

            /* rzr path batch normalize */
            keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

            /* Sampling verification: first, last, every 256th point */
            int all_pass = 1;
            uint8_t cur_privkey[32];
            memcpy(cur_privkey, base_privkey, 32);

            for (int b = 0; b < N; b++) {
                /* Only verify sampled points */
                int do_check = (b == 0) || (b == N - 1) || (b % 256 == 0);
                if (!do_check) {
                    /* Non-sampled points only advance privkey */
                    secp256k1_ec_seckey_tweak_add(secp_ctx, cur_privkey, tweak_bytes);
                    continue;
                }

                uint8_t batch_comp[33], api_comp[33];
                keygen_ge_to_pubkey_bytes(&ge_batch[b], batch_comp, NULL);

                secp256k1_pubkey pubkey;
                secp256k1_ec_pubkey_create(secp_ctx, &pubkey, cur_privkey);
                size_t len = 33;
                secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                              SECP256K1_EC_COMPRESSED);

                if (memcmp(batch_comp, api_comp, 33) != 0) {
                    char bh[67], ah[67];
                    bytes_to_hex_helper(batch_comp, 33, bh);
                    bytes_to_hex_helper(api_comp,   33, ah);
                    printf("  [FAIL] 6.4 b=%d: batch=%s api=%s\n", b, bh, ah);
                    all_pass = 0;
                    fail_count++;
                    break;
                }

                if (b < N - 1)
                    secp256k1_ec_seckey_tweak_add(secp_ctx, cur_privkey, tweak_bytes);
            }
            if (all_pass) {
                printf("  [PASS] 6.4 full batch 4096 points sampling verification (first/last/every 256) all consistent with standard API\n");
                pass_count++;
            }

            free(gej_batch); free(ge_batch); free(rzr_batch);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.5  scalar overflow boundary detection
     *
     *   secp256k1 curve order n (32 bytes big-endian):
     *     FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
     *
     *   Test scenarios:
     *     a) base = n-1 (max valid privkey), verify keygen_privkey_to_gej succeeds
     *     b) base = n (equals order), verify scalar_set_b32 returns overflow=1
     *     c) base = n-2, after two +1 steps scalar should be zero (overflow),
     *        verify inner_overflow detection logic in search_key
     * ------------------------------------------------------------------ */
    {
        /* secp256k1 curve order n */
        static const uint8_t curve_n[32] = {
            0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
            0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
            0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
            0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
        };

        /* 6.5a: base = n-1, keygen_privkey_to_gej should succeed */
        {
            uint8_t privkey_nm1[32];
            memcpy(privkey_nm1, curve_n, 32);
            /* n-1: lowest byte -1 */
            privkey_nm1[31] -= 1;

            secp256k1_gej gej;
            int ret = keygen_privkey_to_gej(secp_ctx, privkey_nm1, &gej);
            if (ret == 0) {
                printf("  [PASS] 6.5a base=n-1: keygen_privkey_to_gej succeeded (valid privkey)\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5a base=n-1: keygen_privkey_to_gej unexpectedly failed\n");
                fail_count++;
            }
        }

        /* 6.5b: base = n, scalar_set_b32 should return overflow=1 */
        {
            secp256k1_scalar s;
            int overflow = 0;
            secp256k1_scalar_set_b32(&s, curve_n, &overflow);
            if (overflow) {
                printf("  [PASS] 6.5b base=n: scalar_set_b32 correctly detected overflow\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5b base=n: scalar_set_b32 did not detect overflow\n");
                fail_count++;
            }
        }

        /* 6.5c: base = n-2, after two +1 steps scalar should be zero
         *   step 1: n-2 + 1 = n-1 (non-zero, normal)
         *   step 2: n-1 + 1 = n ≡ 0 (mod n), scalar_is_zero should be true
         *   this corresponds to the trigger condition for inner_overflow detection in search_key
         */
        {
            uint8_t privkey_nm2[32];
            memcpy(privkey_nm2, curve_n, 32);
            privkey_nm2[31] -= 2;  /* n-2 */

            secp256k1_scalar s;
            int overflow = 0;
            secp256k1_scalar_set_b32(&s, privkey_nm2, &overflow);

            /* step 1: +1, should be non-zero */
            secp256k1_scalar_add(&s, &s, &tweak_scalar);
            int step1_zero = secp256k1_scalar_is_zero(&s);

            /* step 2: +1, should be zero */
            secp256k1_scalar_add(&s, &s, &tweak_scalar);
            int step2_zero = secp256k1_scalar_is_zero(&s);

            if (!overflow && !step1_zero && step2_zero) {
                printf("  [PASS] 6.5c base=n-2: scalar zeroed after two steps, inner_overflow detection logic correct\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5c base=n-2: overflow=%d step1_zero=%d step2_zero=%d (expected 0,0,1)\n",
                       overflow, step1_zero, step2_zero);
                fail_count++;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.6  Uncompressed pubkey path: ge_batch uncompressed bytes consistent with standard API
     *
     *   Verify uncompressed output (65 bytes) of keygen_ge_to_pubkey_bytes
     *   is fully consistent with secp256k1_ec_pubkey_serialize(UNCOMPRESSED).
     *   Test three positions: b=0, b=7, b=15.
     * ------------------------------------------------------------------ */
    {
        const int N = 16;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 200;  /* base = 200 */

        secp256k1_gej gej_batch[16];
        secp256k1_ge  ge_batch[16];
        secp256k1_fe  rzr_batch[16];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        int test_positions[] = {0, 7, 15};
        const char *pos_names[] = {"b=0", "b=7", "b=15"};
        int all_pass = 1;

        for (int t = 0; t < 3; t++) {
            int b = test_positions[t];

            /* keygen path uncompressed pubkey */
            uint8_t keygen_uncomp[65];
            keygen_ge_to_pubkey_bytes(&ge_batch[b], NULL, keygen_uncomp);

            /* standard API uncompressed pubkey (privkey = base + b) */
            uint8_t privkey_b[32] = {0};
            privkey_b[31] = (uint8_t)(200 + b);
            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey_b);
            uint8_t api_uncomp[65];
            size_t len = 65;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len, &pubkey,
                                          SECP256K1_EC_UNCOMPRESSED);

            if (memcmp(keygen_uncomp, api_uncomp, 65) != 0) {
                char kh[131], ah[131];
                bytes_to_hex_helper(keygen_uncomp, 65, kh);
                bytes_to_hex_helper(api_uncomp,    65, ah);
                printf("  [FAIL] 6.6 uncompressed pubkey %s:\n"
                       "         keygen: %s\n"
                       "         api:    %s\n",
                       pos_names[t], kh, ah);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.6 uncompressed pubkey path (b=0/7/15) fully consistent with standard API\n");
            pass_count++;
        }
    }
}


#if defined(__AVX512IFMA__) && !defined(USE_PUBKEY_API_ONLY)
/* ------------------------------------------------------------------
 * test_gej_add_ge_var_16way
 *
 * Verify gej_add_ge_var_16way / gej_add_ge_var_16way_normed in search_key
 * functional correctness: 16-way parallel point addition results consistent with standard secp256k1_gej_add_ge_var.
 *
 * Coverage scenarios:
 *   8.1  Single-step verification: each of 16 chains advances 1 step, AVX-512 results vs scalar
 *   8.2  Multi-step continuous advance (simulate search_key inner loop, alternating
 *        gej_add_ge_var_16way / gej_add_ge_var_16way_normed）：
 *        normalize after N steps, each point pubkey fully consistent with standard API
 *   8.3  Edge cases: a=b (point doubling) and a=infinity handling
 * ------------------------------------------------------------------ */
static void test_gej_add_ge_var_16way(void) {
    printf("\n=== gej_add_ge_var_16way functional correctness tests ===\n");

    /* ------------------------------------------------------------------ */
    /* 8.1  Single-step verification: each of 16 chains advances 1 step      */
    /* ------------------------------------------------------------------ */
    {
        printf("  [8.1 single-step verification: 16-way parallel point addition vs scalar]\n");
        int all_pass = 1;

        int base_vals[16] = {3, 7, 11, 17, 23, 31, 37, 41,
                             43, 47, 53, 59, 61, 67, 71, 73};
        secp256k1_gej chain_gej[16];

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32] = {0};
            pk[31] = (uint8_t)base_vals[ch];
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        secp256k1_gej avx512_next[16];
        secp256k1_fe  avx512_rzr[16];
        gej_add_ge_var_16way(avx512_next, chain_gej, &G_affine, avx512_rzr, 0);

        secp256k1_gej ref_next[16];
        secp256k1_fe  ref_rzr[16];
        for (int ch = 0; ch < 16; ch++)
            secp256k1_gej_add_ge_var(&ref_next[ch], &chain_gej[ch], &G_affine, &ref_rzr[ch]);

        secp256k1_ge avx512_ge[16], ref_ge[16];
        keygen_batch_normalize(avx512_next, avx512_ge, 16);
        keygen_batch_normalize(ref_next,    ref_ge,    16);

        for (int ch = 0; ch < 16; ch++) {
            uint8_t avx512_comp[33], ref_comp[33];
            keygen_ge_to_pubkey_bytes(&avx512_ge[ch], avx512_comp, NULL);
            keygen_ge_to_pubkey_bytes(&ref_ge[ch],    ref_comp,    NULL);
            if (memcmp(avx512_comp, ref_comp, 33) != 0) {
                char avx512_hex[67], ref_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(ref_comp,    33, ref_hex);
                printf("  [FAIL] 8.1 lane%d（base=%d+1）: AVX512=%s REF=%s\n",
                       ch, base_vals[ch], avx512_hex, ref_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 8.1 single-step verification: 16-way parallel point addition fully consistent with scalar\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.2  Multi-step advance (alternating 16way / 16way_normed)            */
    /* ------------------------------------------------------------------ */
    {
        const int STEPS = 200;
        printf("  [8.2 multi-step advance: compare with standard API after %d steps (alternating 16way/16way_normed)]\n", STEPS);
        int all_pass = 1;

        uint32_t base_vals[16] = {
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
            9000,10000,11000,12000,13000,14000,15000,16000
        };
        secp256k1_gej chain_gej[16];

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32] = {0};
            pk[29] = (uint8_t)(base_vals[ch] >> 16);
            pk[30] = (uint8_t)(base_vals[ch] >> 8);
            pk[31] = (uint8_t)(base_vals[ch]);
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        /* Simulate search_key: step=0 uses normed=0, step>0 uses normed=1 */
        for (int step = 0; step < STEPS; step++) {
            secp256k1_gej next_gej[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way(next_gej, chain_gej, &G_affine, rzr,
                                 step == 0 ? 0 : 1);
            for (int ch = 0; ch < 16; ch++)
                chain_gej[ch] = next_gej[ch];
        }

        secp256k1_ge chain_ge[16];
        keygen_batch_normalize(chain_gej, chain_ge, 16);

        for (int ch = 0; ch < 16; ch++) {
            uint32_t expected_k = base_vals[ch] + (uint32_t)STEPS;
            uint8_t  expected_privkey[32] = {0};
            expected_privkey[29] = (uint8_t)(expected_k >> 16);
            expected_privkey[30] = (uint8_t)(expected_k >> 8);
            expected_privkey[31] = (uint8_t)(expected_k);

            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, expected_privkey);
            uint8_t api_comp[33];
            size_t  api_len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &api_len,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);

            uint8_t avx512_comp[33];
            keygen_ge_to_pubkey_bytes(&chain_ge[ch], avx512_comp, NULL);

            if (memcmp(avx512_comp, api_comp, 33) != 0) {
                char avx512_hex[67], api_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(api_comp,    33, api_hex);
                printf("  [FAIL] 8.2 lane%d (base=%u + %d steps): AVX512=%s API=%s\n",
                       ch, base_vals[ch], STEPS, avx512_hex, api_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 8.2 multi-step advance: 16 chains fully consistent with standard API after %d steps\n", STEPS);
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.3  Edge cases: a=b (point doubling) and a=infinity                  */
    /* ------------------------------------------------------------------ */
    {
        printf("  [8.3 edge cases: point doubling (a=b) and point at infinity (a=infinity)]\n");
        int all_pass = 1;

        /* 8.3a: lane0 = 1*G = b, expected result = 2*G */
        {
            secp256k1_gej chain_gej[16];
            uint8_t privkey_1[32] = {0}; privkey_1[31] = 1;
            keygen_privkey_to_gej(secp_ctx, privkey_1, &chain_gej[0]);
            for (int ch = 1; ch < 16; ch++) {
                uint8_t pk[32] = {0};
                pk[31] = (uint8_t)(10 + ch);
                keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
            }

            secp256k1_gej next_gej[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way(next_gej, chain_gej, &G_affine, rzr, 0);

            secp256k1_ge next_ge[16];
            keygen_batch_normalize(next_gej, next_ge, 16);

            uint8_t privkey_2[32] = {0}; privkey_2[31] = 2;
            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, privkey_2);
            uint8_t api_comp[33];
            size_t  api_len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &api_len,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);

            uint8_t avx512_comp[33];
            keygen_ge_to_pubkey_bytes(&next_ge[0], avx512_comp, NULL);

            if (memcmp(avx512_comp, api_comp, 33) != 0) {
                char avx512_hex[67], api_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(api_comp,    33, api_hex);
                printf("  [FAIL] 8.3a point doubling (a=1*G, b=G): AVX512=%s expected=%s\n",
                       avx512_hex, api_hex);
                all_pass = 0;
                fail_count++;
            }
        }

        /* 8.3b: lane0 = infinity, expected result = G (privkey=1) */
        {
            secp256k1_gej chain_gej[16];
            secp256k1_gej_set_infinity(&chain_gej[0]);
            for (int ch = 1; ch < 16; ch++) {
                uint8_t pk[32] = {0};
                pk[31] = (uint8_t)(20 + ch);
                keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
            }

            secp256k1_gej next_gej[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way(next_gej, chain_gej, &G_affine, rzr, 0);

            secp256k1_ge next_ge[16];
            keygen_batch_normalize(next_gej, next_ge, 16);

            uint8_t privkey_1[32] = {0}; privkey_1[31] = 1;
            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, privkey_1);
            uint8_t api_comp[33];
            size_t  api_len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &api_len,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);

            uint8_t avx512_comp[33];
            keygen_ge_to_pubkey_bytes(&next_ge[0], avx512_comp, NULL);

            if (memcmp(avx512_comp, api_comp, 33) != 0) {
                char avx512_hex[67], api_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(api_comp,    33, api_hex);
                printf("  [FAIL] 8.3b point at infinity (a=infinity, b=G): AVX512=%s expected=%s\n",
                       avx512_hex, api_hex);
                all_pass = 0;
                fail_count++;
            }
        }

        if (all_pass) {
            printf("  [PASS] 8.3 edge cases: point doubling and point at infinity handled correctly\n");
            pass_count++;
        }
    }
}

/*
 * Test adapter: wraps gej_add_ge_var_16way_soa's new scatter-pointer interface
 * to accept a contiguous r_aos[16] array (for backward compatibility in tests).
 */
static void gej_add_ge_var_16way_soa_compat(secp256k1_gej_16x *r_soa,
                                             const secp256k1_gej_16x *a_soa,
                                             const secp256k1_ge *b,
                                             secp256k1_gej r_aos[16],
                                             secp256k1_fe rzr_out[16],
                                             int normed)
{
    if (r_aos != NULL) {
        secp256k1_gej *ptrs[16];
        for (int i = 0; i < 16; i++) ptrs[i] = &r_aos[i];
        gej_add_ge_var_16way_soa(r_soa, a_soa, b, ptrs, rzr_out, normed);
    } else {
        gej_add_ge_var_16way_soa(r_soa, a_soa, b, NULL, rzr_out, normed);
    }
}

/* ------------------------------------------------------------------
 * test_gej_add_ge_var_16way_soa
 *
 * Verify functional correctness of gej_add_ge_var_16way_soa (SoA persistent version).
 * Coverage scenarios:
 *   8.4  Single-step SoA correctness: SoA output vs scalar reference
 *   8.5  Multi-step SoA continuous advance (simulating keysearch hot loop)
 *   8.6  Edge cases: point doubling (a=b) and point at infinity via SoA
 *   8.7  SoA vs AoS bit-exact consistency
 * ------------------------------------------------------------------ */
static void test_gej_add_ge_var_16way_soa(void) {
    printf("\n=== gej_add_ge_var_16way_soa functional correctness tests ===\n");

    /* ------------------------------------------------------------------ */
    /* 8.4  Single-step SoA correctness: 16-way parallel point addition      */
    /*      vs scalar secp256k1_gej_add_ge_var                               */
    /* ------------------------------------------------------------------ */
    {
        printf("  [8.4 single-step SoA verification: 16-way parallel point addition vs scalar]\n");
        int all_pass = 1;

        int base_vals[16] = {3, 7, 11, 17, 23, 31, 37, 41,
                             43, 47, 53, 59, 61, 67, 71, 73};
        secp256k1_gej chain_gej[16];

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32] = {0};
            pk[31] = (uint8_t)base_vals[ch];
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        /* Load AoS -> SoA */
        secp256k1_gej_16x a_soa;
        gej_16x_load(&a_soa, chain_gej);

        /* Call SoA version with normed=0 */
        secp256k1_gej_16x r_soa;
        secp256k1_gej r_aos[16];
        secp256k1_fe  rzr_out[16];
        gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, r_aos, rzr_out, 0);

        /* Scalar reference */
        secp256k1_gej ref_next[16];
        secp256k1_fe  ref_rzr[16];
        for (int ch = 0; ch < 16; ch++)
            secp256k1_gej_add_ge_var(&ref_next[ch], &chain_gej[ch], &G_affine, &ref_rzr[ch]);

        /* Verification 1: r_aos compressed pubkey vs scalar reference */
        secp256k1_ge avx512_ge[16], ref_ge[16];
        keygen_batch_normalize(r_aos,    avx512_ge, 16);
        keygen_batch_normalize(ref_next, ref_ge,    16);

        for (int ch = 0; ch < 16; ch++) {
            uint8_t avx512_comp[33], ref_comp[33];
            keygen_ge_to_pubkey_bytes(&avx512_ge[ch], avx512_comp, NULL);
            keygen_ge_to_pubkey_bytes(&ref_ge[ch],    ref_comp,    NULL);
            if (memcmp(avx512_comp, ref_comp, 33) != 0) {
                char avx512_hex[67], ref_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(ref_comp,    33, ref_hex);
                printf("  [FAIL] 8.4a lane%d (base=%d+1): SoA_AoS=%s REF=%s\n",
                       ch, base_vals[ch], avx512_hex, ref_hex);
                all_pass = 0;
                fail_count++;
            }
        }

        /* Verification 2: r_soa -> gej_16x_store vs r_aos, bit-exact comparison */
        secp256k1_gej r_from_soa[16];
        gej_16x_store(r_from_soa, &r_soa);

        int soa_aos_match = 1;
        for (int ch = 0; ch < 16; ch++) {
            secp256k1_fe fx1, fy1, fz1, fx2, fy2, fz2;
            fx1 = r_from_soa[ch].x; secp256k1_fe_normalize(&fx1);
            fy1 = r_from_soa[ch].y; secp256k1_fe_normalize(&fy1);
            fz1 = r_from_soa[ch].z; secp256k1_fe_normalize(&fz1);
            fx2 = r_aos[ch].x; secp256k1_fe_normalize(&fx2);
            fy2 = r_aos[ch].y; secp256k1_fe_normalize(&fy2);
            fz2 = r_aos[ch].z; secp256k1_fe_normalize(&fz2);

            if (!secp256k1_fe_equal(&fx1, &fx2) ||
                !secp256k1_fe_equal(&fy1, &fy2) ||
                !secp256k1_fe_equal(&fz1, &fz2)) {
                printf("  [FAIL] 8.4b lane%d (base=%d): SoA->AoS mismatch with direct r_aos\n",
                       ch, base_vals[ch]);
                soa_aos_match = 0;
                all_pass = 0;
                fail_count++;
            }
        }
        if (soa_aos_match) {
            printf("  [PASS] 8.4b SoA->AoS output matches direct r_aos for all 16 lanes\n");
            pass_count++;
        }

        if (all_pass) {
            printf("  [PASS] 8.4 single-step SoA verification: fully consistent with scalar\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.5  Multi-step SoA continuous advance: simulating keysearch hot loop  */
    /*      step=0 normed=0, step>0 normed=1, 200 steps total                */
    /* ------------------------------------------------------------------ */
    {
        const int STEPS = 200;
        printf("  [8.5 multi-step SoA advance: compare with standard API after %d steps]\n", STEPS);
        int all_pass = 1;

        uint32_t base_vals[16] = {
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
            9000,10000,11000,12000,13000,14000,15000,16000
        };
        secp256k1_gej chain_gej[16];

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32] = {0};
            pk[29] = (uint8_t)(base_vals[ch] >> 16);
            pk[30] = (uint8_t)(base_vals[ch] >> 8);
            pk[31] = (uint8_t)(base_vals[ch]);
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        /* Load initial AoS -> SoA */
        secp256k1_gej_16x a_soa;
        gej_16x_load(&a_soa, chain_gej);

        /* Simulate keysearch hot loop: step=0 normed=0, step>0 normed=1 */
        int soa_aos_mismatch_count = 0;
        for (int step = 0; step < STEPS; step++) {
            secp256k1_gej_16x r_soa;
            secp256k1_gej r_aos[16];
            secp256k1_fe  rzr[16];
            int normed = (step == 0) ? 0 : 1;
            gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, r_aos, rzr, normed);

            /* Every 50 steps: verify SoA->AoS matches r_aos */
            if (step % 50 == 0 || step == STEPS - 1) {
                secp256k1_gej r_from_soa[16];
                gej_16x_store(r_from_soa, &r_soa);
                for (int ch = 0; ch < 16; ch++) {
                    secp256k1_fe fx1, fy1, fz1, fx2, fy2, fz2;
                    fx1 = r_from_soa[ch].x; secp256k1_fe_normalize(&fx1);
                    fy1 = r_from_soa[ch].y; secp256k1_fe_normalize(&fy1);
                    fz1 = r_from_soa[ch].z; secp256k1_fe_normalize(&fz1);
                    fx2 = r_aos[ch].x; secp256k1_fe_normalize(&fx2);
                    fy2 = r_aos[ch].y; secp256k1_fe_normalize(&fy2);
                    fz2 = r_aos[ch].z; secp256k1_fe_normalize(&fz2);
                    if (!secp256k1_fe_equal(&fx1, &fx2) ||
                        !secp256k1_fe_equal(&fy1, &fy2) ||
                        !secp256k1_fe_equal(&fz1, &fz2)) {
                        if (soa_aos_mismatch_count == 0) {
                            printf("  [FAIL] 8.5b SoA->AoS mismatch at step=%d lane%d\n",
                                   step, ch);
                            fail_count++;
                        }
                        soa_aos_mismatch_count++;
                        all_pass = 0;
                    }
                }
            }

            /* Pass SoA output as next input */
            a_soa = r_soa;
        }

        if (soa_aos_mismatch_count == 0) {
            printf("  [PASS] 8.5b SoA->AoS consistency at sampled steps for all lanes\n");
            pass_count++;
        } else if (soa_aos_mismatch_count > 1) {
            printf("  [INFO] 8.5b total SoA->AoS mismatches: %d (only first reported)\n",
                   soa_aos_mismatch_count);
        }

        /* Verification 1: final pubkey vs API */
        /* Need to get final AoS from SoA for batch_normalize */
        secp256k1_gej final_aos[16];
        gej_16x_store(final_aos, &a_soa);

        secp256k1_ge chain_ge[16];
        keygen_batch_normalize(final_aos, chain_ge, 16);

        for (int ch = 0; ch < 16; ch++) {
            uint32_t expected_k = base_vals[ch] + (uint32_t)STEPS;
            uint8_t  expected_privkey[32] = {0};
            expected_privkey[29] = (uint8_t)(expected_k >> 16);
            expected_privkey[30] = (uint8_t)(expected_k >> 8);
            expected_privkey[31] = (uint8_t)(expected_k);

            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, expected_privkey);
            uint8_t api_comp[33];
            size_t  api_len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &api_len,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);

            uint8_t avx512_comp[33];
            keygen_ge_to_pubkey_bytes(&chain_ge[ch], avx512_comp, NULL);

            if (memcmp(avx512_comp, api_comp, 33) != 0) {
                char avx512_hex[67], api_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(api_comp,    33, api_hex);
                printf("  [FAIL] 8.5a lane%d (base=%u + %d steps): SoA=%s API=%s\n",
                       ch, base_vals[ch], STEPS, avx512_hex, api_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 8.5 multi-step SoA advance: 16 chains fully consistent with standard API after %d steps\n", STEPS);
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.6  Edge cases: point doubling (a=b) via SoA                           */
    /* ------------------------------------------------------------------ */
    {
        printf("  [8.6 edge cases: SoA point doubling (a=b)]\n");
        int all_pass = 1;

        /* 8.6a: lane0 = 1*G = b (triggers point doubling h=0), rest are normal */
        {
            secp256k1_gej chain_gej[16];
            uint8_t privkey_1[32] = {0}; privkey_1[31] = 1;
            keygen_privkey_to_gej(secp_ctx, privkey_1, &chain_gej[0]);
            for (int ch = 1; ch < 16; ch++) {
                uint8_t pk[32] = {0};
                pk[31] = (uint8_t)(10 + ch);
                keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
            }

            secp256k1_gej_16x a_soa;
            gej_16x_load(&a_soa, chain_gej);

            secp256k1_gej_16x r_soa;
            secp256k1_gej r_aos[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, r_aos, rzr, 0);

            secp256k1_ge next_ge[16];
            keygen_batch_normalize(r_aos, next_ge, 16);

            /* lane0 should be 2*G */
            uint8_t privkey_2[32] = {0}; privkey_2[31] = 2;
            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, privkey_2);
            uint8_t api_comp[33];
            size_t  api_len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &api_len,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);

            uint8_t avx512_comp[33];
            keygen_ge_to_pubkey_bytes(&next_ge[0], avx512_comp, NULL);

            if (memcmp(avx512_comp, api_comp, 33) != 0) {
                char avx512_hex[67], api_hex[67];
                bytes_to_hex_helper(avx512_comp, 33, avx512_hex);
                bytes_to_hex_helper(api_comp,    33, api_hex);
                printf("  [FAIL] 8.6a point doubling (a=1*G, b=G): SoA=%s expected=%s\n",
                       avx512_hex, api_hex);
                all_pass = 0;
                fail_count++;
            }

            /* Verify remaining lanes are still correct */
            for (int ch = 1; ch < 16; ch++) {
                uint32_t expected_k = 10 + ch + 1;
                uint8_t  epk[32] = {0};
                epk[31] = (uint8_t)(expected_k);

                secp256k1_pubkey ep;
                secp256k1_ec_pubkey_create(secp_ctx, &ep, epk);
                uint8_t ec[33];
                size_t  el = 33;
                secp256k1_ec_pubkey_serialize(secp_ctx, ec, &el,
                                              &ep, SECP256K1_EC_COMPRESSED);

                uint8_t ac[33];
                keygen_ge_to_pubkey_bytes(&next_ge[ch], ac, NULL);
                if (memcmp(ac, ec, 33) != 0) {
                    char ah[67], eh[67];
                    bytes_to_hex_helper(ac, 33, ah);
                    bytes_to_hex_helper(ec, 33, eh);
                    printf("  [FAIL] 8.6a lane%d (base=%d+1): SoA=%s expected=%s\n",
                           ch, 10 + ch, ah, eh);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }

        /* 8.6b: multiple lanes trigger point doubling simultaneously */
        /*        lane0=1*G, lane4=1*G, lane8=1*G, lane12=1*G, rest normal */
        {
            secp256k1_gej chain_gej[16];
            int doubling_lanes[4] = {0, 4, 8, 12};
            for (int ch = 0; ch < 16; ch++) {
                uint8_t pk[32] = {0};
                int is_doubling = 0;
                for (int d = 0; d < 4; d++) {
                    if (ch == doubling_lanes[d]) { is_doubling = 1; break; }
                }
                if (is_doubling) {
                    pk[31] = 1; /* 1*G => triggers doubling since b=G */
                } else {
                    pk[31] = (uint8_t)(50 + ch);
                }
                keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
            }

            secp256k1_gej_16x a_soa;
            gej_16x_load(&a_soa, chain_gej);

            secp256k1_gej_16x r_soa;
            secp256k1_gej r_aos[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, r_aos, rzr, 0);

            secp256k1_ge next_ge[16];
            keygen_batch_normalize(r_aos, next_ge, 16);

            for (int ch = 0; ch < 16; ch++) {
                int is_doubling = 0;
                for (int d = 0; d < 4; d++) {
                    if (ch == doubling_lanes[d]) { is_doubling = 1; break; }
                }
                uint32_t expected_k = is_doubling ? 2 : (uint32_t)(50 + ch + 1);
                uint8_t  epk[32] = {0};
                epk[31] = (uint8_t)(expected_k);

                secp256k1_pubkey ep;
                secp256k1_ec_pubkey_create(secp_ctx, &ep, epk);
                uint8_t ec[33]; size_t el = 33;
                secp256k1_ec_pubkey_serialize(secp_ctx, ec, &el, &ep, SECP256K1_EC_COMPRESSED);

                uint8_t ac[33];
                keygen_ge_to_pubkey_bytes(&next_ge[ch], ac, NULL);
                if (memcmp(ac, ec, 33) != 0) {
                    char ah[67], eh[67];
                    bytes_to_hex_helper(ac, 33, ah);
                    bytes_to_hex_helper(ec, 33, eh);
                    printf("  [FAIL] 8.6b lane%d (%s, expected_k=%u): SoA=%s expected=%s\n",
                           ch, is_doubling ? "doubling" : "normal",
                           expected_k, ah, eh);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }

        /* 8.6c: all 16 lanes trigger point doubling (worst case: all a=1*G) */
        {
            secp256k1_gej chain_gej[16];
            uint8_t privkey_1[32] = {0}; privkey_1[31] = 1;
            for (int ch = 0; ch < 16; ch++) {
                keygen_privkey_to_gej(secp_ctx, privkey_1, &chain_gej[ch]);
            }

            secp256k1_gej_16x a_soa;
            gej_16x_load(&a_soa, chain_gej);

            secp256k1_gej_16x r_soa;
            secp256k1_gej r_aos[16];
            secp256k1_fe  rzr[16];
            gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, r_aos, rzr, 0);

            secp256k1_ge next_ge[16];
            keygen_batch_normalize(r_aos, next_ge, 16);

            /* All lanes should be 2*G */
            uint8_t privkey_2[32] = {0}; privkey_2[31] = 2;
            secp256k1_pubkey ep;
            secp256k1_ec_pubkey_create(secp_ctx, &ep, privkey_2);
            uint8_t ec[33]; size_t el = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, ec, &el, &ep, SECP256K1_EC_COMPRESSED);

            for (int ch = 0; ch < 16; ch++) {
                uint8_t ac[33];
                keygen_ge_to_pubkey_bytes(&next_ge[ch], ac, NULL);
                if (memcmp(ac, ec, 33) != 0) {
                    char ah[67], eh[67];
                    bytes_to_hex_helper(ac, 33, ah);
                    bytes_to_hex_helper(ec, 33, eh);
                    printf("  [FAIL] 8.6c all-doubling lane%d: SoA=%s expected=%s\n",
                           ch, ah, eh);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }

        if (all_pass) {
            printf("  [PASS] 8.6 edge cases: SoA point doubling handled correctly\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.7  SoA vs AoS bit-exact consistency                                 */
    /* ------------------------------------------------------------------ */
    {
        printf("  [8.7 SoA vs AoS bit-exact consistency]\n");
        int all_pass = 1;

        int base_vals[16] = {3, 7, 11, 17, 23, 31, 37, 41,
                             43, 47, 53, 59, 61, 67, 71, 73};
        secp256k1_gej chain_gej[16];

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32] = {0};
            pk[31] = (uint8_t)base_vals[ch];
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        /* --- Verification 1: normed=0 --- */
        {
            /* AoS version */
            secp256k1_gej aos_result[16];
            secp256k1_fe  aos_rzr[16];
            gej_add_ge_var_16way(aos_result, chain_gej, &G_affine, aos_rzr, 0);

            /* SoA version */
            secp256k1_gej_16x a_soa;
            gej_16x_load(&a_soa, chain_gej);
            secp256k1_gej_16x r_soa;
            secp256k1_gej soa_result[16];
            secp256k1_fe  soa_rzr[16];
            gej_add_ge_var_16way_soa_compat(&r_soa, &a_soa, &G_affine, soa_result, soa_rzr, 0);

            for (int ch = 0; ch < 16; ch++) {
                secp256k1_fe ax1, ay1, az1, ax2, ay2, az2;
                ax1 = aos_result[ch].x; secp256k1_fe_normalize(&ax1);
                ay1 = aos_result[ch].y; secp256k1_fe_normalize(&ay1);
                az1 = aos_result[ch].z; secp256k1_fe_normalize(&az1);
                ax2 = soa_result[ch].x; secp256k1_fe_normalize(&ax2);
                ay2 = soa_result[ch].y; secp256k1_fe_normalize(&ay2);
                az2 = soa_result[ch].z; secp256k1_fe_normalize(&az2);

                if (!secp256k1_fe_equal(&ax1, &ax2) ||
                    !secp256k1_fe_equal(&ay1, &ay2) ||
                    !secp256k1_fe_equal(&az1, &az2)) {
                    printf("  [FAIL] 8.7a lane%d (base=%d, normed=0): r_aos x/y/z mismatch between AoS and SoA\n",
                           ch, base_vals[ch]);
                    all_pass = 0;
                    fail_count++;
                }

                /* Compare rzr */
                secp256k1_fe r1 = aos_rzr[ch], r2 = soa_rzr[ch];
                secp256k1_fe_normalize(&r1);
                secp256k1_fe_normalize(&r2);
                if (!secp256k1_fe_equal(&r1, &r2)) {
                    printf("  [FAIL] 8.7a lane%d (base=%d, normed=0): rzr mismatch between AoS and SoA\n",
                           ch, base_vals[ch]);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }

        /* --- Verification 2: normed=1 (requires one step of normed=0 first) --- */
        {
            /* First step: normed=0 to produce normalized output as input for normed=1 */
            secp256k1_gej aos_step1[16], soa_step1[16];
            secp256k1_fe  aos_rzr1[16], soa_rzr1[16];

            gej_add_ge_var_16way(aos_step1, chain_gej, &G_affine, aos_rzr1, 0);

            secp256k1_gej_16x a_soa, r_soa_step1;
            gej_16x_load(&a_soa, chain_gej);
            gej_add_ge_var_16way_soa_compat(&r_soa_step1, &a_soa, &G_affine, soa_step1, soa_rzr1, 0);

            /* Second step: normed=1 */
            secp256k1_gej aos_step2[16], soa_step2[16];
            secp256k1_fe  aos_rzr2[16], soa_rzr2[16];

            gej_add_ge_var_16way(aos_step2, aos_step1, &G_affine, aos_rzr2, 1);

            secp256k1_gej_16x r_soa_step2;
            gej_add_ge_var_16way_soa_compat(&r_soa_step2, &r_soa_step1, &G_affine, soa_step2, soa_rzr2, 1);

            for (int ch = 0; ch < 16; ch++) {
                secp256k1_fe ax1, ay1, az1, ax2, ay2, az2;
                ax1 = aos_step2[ch].x; secp256k1_fe_normalize(&ax1);
                ay1 = aos_step2[ch].y; secp256k1_fe_normalize(&ay1);
                az1 = aos_step2[ch].z; secp256k1_fe_normalize(&az1);
                ax2 = soa_step2[ch].x; secp256k1_fe_normalize(&ax2);
                ay2 = soa_step2[ch].y; secp256k1_fe_normalize(&ay2);
                az2 = soa_step2[ch].z; secp256k1_fe_normalize(&az2);

                if (!secp256k1_fe_equal(&ax1, &ax2) ||
                    !secp256k1_fe_equal(&ay1, &ay2) ||
                    !secp256k1_fe_equal(&az1, &az2)) {
                    printf("  [FAIL] 8.7b lane%d (base=%d, normed=1): r_aos x/y/z mismatch between AoS and SoA\n",
                           ch, base_vals[ch]);
                    all_pass = 0;
                    fail_count++;
                }

                /* Compare rzr */
                secp256k1_fe r1 = aos_rzr2[ch], r2 = soa_rzr2[ch];
                secp256k1_fe_normalize(&r1);
                secp256k1_fe_normalize(&r2);
                if (!secp256k1_fe_equal(&r1, &r2)) {
                    printf("  [FAIL] 8.7b lane%d (base=%d, normed=1): rzr mismatch between AoS and SoA\n",
                           ch, base_vals[ch]);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }

        if (all_pass) {
            printf("  [PASS] 8.7 SoA vs AoS bit-exact consistency: fully matched\n");
            pass_count++;
        }
    }
}

/* ------------------------------------------------------------------
 * test_keygen_ge_to_pubkey_bytes_16way
 *
 * Verify keygen_ge_to_pubkey_bytes_16way (AVX-512 batch pubkey byte conversion)
 * produces bit-exact results compared with scalar keygen_ge_to_pubkey_bytes and
 * the secp256k1 public API.
 *
 * Coverage scenarios:
 *   8.8a  Compressed format: 16way vs scalar, 33-byte compressed pubkey bit-exact
 *   8.8b  Uncompressed format: 16way vs scalar, 65-byte uncompressed pubkey bit-exact
 *   8.8c  Public API comparison: 16way output vs secp256k1_ec_pubkey_serialize
 *   8.8d  NULL pointer skip: compressed_out or uncompressed_out partially NULL
 * ------------------------------------------------------------------ */
static void test_keygen_ge_to_pubkey_bytes_16way(void) {
    printf("\n=== keygen_ge_to_pubkey_bytes_16way tests ===\n");

    /* Use 16 distinct known private keys to generate ge points */
    static const int base_keys[16] = {
        3, 7, 11, 17, 23, 31, 37, 41,
        43, 47, 53, 59, 61, 67, 71, 73
    };

    secp256k1_scalar tweak;
    secp256k1_scalar_set_int(&tweak, 1);

    secp256k1_ge ge_points[16];
    secp256k1_scalar scalars[16];

    /* Generate 16 ge points */
    for (int i = 0; i < 16; i++) {
        secp256k1_scalar s;
        secp256k1_scalar_set_int(&s, (unsigned int)base_keys[i]);
        scalars[i] = s;

        uint8_t pk[32];
        secp256k1_scalar_get_b32(pk, &s);

        secp256k1_gej gej;
        keygen_privkey_to_gej(secp_ctx, pk, &gej);

        /* Single-point normalization */
        secp256k1_ge_set_gej(&ge_points[i], &gej);
    }

    /* ---- 8.8a compressed format: 16way vs scalar ---- */
    {
        int pass = 1;
        uint8_t comp_16way[16][33];
        uint8_t uncomp_16way[16][65];
        uint8_t *comp_ptrs[16], *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) {
            comp_ptrs[i] = comp_16way[i];
            uncomp_ptrs[i] = uncomp_16way[i];
        }
        keygen_ge_to_pubkey_bytes_16way(ge_points, comp_ptrs, uncomp_ptrs);

        for (int i = 0; i < 16; i++) {
            uint8_t comp_scalar[33];
            uint8_t uncomp_scalar[65];
            keygen_ge_to_pubkey_bytes(&ge_points[i], comp_scalar, uncomp_scalar);

            if (memcmp(comp_16way[i], comp_scalar, 33) != 0) {
                char hex16[67], hexsc[67];
                bytes_to_hex_helper(comp_16way[i], 33, hex16);
                bytes_to_hex_helper(comp_scalar, 33, hexsc);
                printf("  [FAIL] 8.8a lane%d compressed: 16way=%s scalar=%s\n",
                       i, hex16, hexsc);
                pass = 0;
                fail_count++;
            }
        }
        if (pass) {
            printf("  [PASS] 8.8a compressed format: 16way vs scalar bit-exact for all 16 lanes\n");
            pass_count++;
        }
    }

    /* ---- 8.8b uncompressed format: 16way vs scalar ---- */
    {
        int pass = 1;
        uint8_t comp_16way[16][33];
        uint8_t uncomp_16way[16][65];
        uint8_t *comp_ptrs[16], *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) {
            comp_ptrs[i] = comp_16way[i];
            uncomp_ptrs[i] = uncomp_16way[i];
        }
        keygen_ge_to_pubkey_bytes_16way(ge_points, comp_ptrs, uncomp_ptrs);

        for (int i = 0; i < 16; i++) {
            uint8_t comp_scalar[33];
            uint8_t uncomp_scalar[65];
            keygen_ge_to_pubkey_bytes(&ge_points[i], comp_scalar, uncomp_scalar);

            if (memcmp(uncomp_16way[i], uncomp_scalar, 65) != 0) {
                char hex16[131], hexsc[131];
                bytes_to_hex_helper(uncomp_16way[i], 65, hex16);
                bytes_to_hex_helper(uncomp_scalar, 65, hexsc);
                printf("  [FAIL] 8.8b lane%d uncompressed: 16way=%s scalar=%s\n",
                       i, hex16, hexsc);
                pass = 0;
                fail_count++;
            }
        }
        if (pass) {
            printf("  [PASS] 8.8b uncompressed format: 16way vs scalar bit-exact for all 16 lanes\n");
            pass_count++;
        }
    }

    /* ---- 8.8c public API comparison: 16way output vs secp256k1_ec_pubkey_serialize ---- */
    {
        int pass = 1;
        uint8_t comp_16way[16][33];
        uint8_t uncomp_16way[16][65];
        uint8_t *comp_ptrs[16], *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) {
            comp_ptrs[i] = comp_16way[i];
            uncomp_ptrs[i] = uncomp_16way[i];
        }
        keygen_ge_to_pubkey_bytes_16way(ge_points, comp_ptrs, uncomp_ptrs);

        for (int i = 0; i < 16; i++) {
            uint8_t pk[32];
            secp256k1_scalar_get_b32(pk, &scalars[i]);

            secp256k1_pubkey api_pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, pk);

            uint8_t api_comp[33], api_uncomp[65];
            size_t len_c = 33, len_u = 65;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len_c,
                                          &api_pubkey, SECP256K1_EC_COMPRESSED);
            secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len_u,
                                          &api_pubkey, SECP256K1_EC_UNCOMPRESSED);

            if (memcmp(comp_16way[i], api_comp, 33) != 0) {
                char hex16[67], hexapi[67];
                bytes_to_hex_helper(comp_16way[i], 33, hex16);
                bytes_to_hex_helper(api_comp, 33, hexapi);
                printf("  [FAIL] 8.8c lane%d compressed vs API: 16way=%s API=%s\n",
                       i, hex16, hexapi);
                pass = 0;
                fail_count++;
            }

            if (memcmp(uncomp_16way[i], api_uncomp, 65) != 0) {
                char hex16[131], hexapi[131];
                bytes_to_hex_helper(uncomp_16way[i], 65, hex16);
                bytes_to_hex_helper(api_uncomp, 65, hexapi);
                printf("  [FAIL] 8.8c lane%d uncompressed vs API: 16way=%s API=%s\n",
                       i, hex16, hexapi);
                pass = 0;
                fail_count++;
            }
        }
        if (pass) {
            printf("  [PASS] 8.8c public API comparison: 16way output matches secp256k1 API for all 16 lanes\n");
            pass_count++;
        }
    }

    /* ---- 8.8d NULL pointer skip test ---- */
    {
        int pass = 1;

        /* Test 1: pass NULL for all compressed_out, only output uncompressed */
        {
            uint8_t uncomp_16way[16][65];
            uint8_t *uncomp_ptrs[16];
            for (int i = 0; i < 16; i++)
                uncomp_ptrs[i] = uncomp_16way[i];
            keygen_ge_to_pubkey_bytes_16way(ge_points, NULL, uncomp_ptrs);

            for (int i = 0; i < 16; i++) {
                uint8_t uncomp_scalar[65];
                keygen_ge_to_pubkey_bytes(&ge_points[i], NULL, uncomp_scalar);
                if (memcmp(uncomp_16way[i], uncomp_scalar, 65) != 0) {
                    printf("  [FAIL] 8.8d NULL compressed: lane%d uncompressed mismatch\n", i);
                    pass = 0;
                    fail_count++;
                }
            }
        }

        /* Test 2: pass NULL for all uncompressed_out, only output compressed */
        {
            uint8_t comp_16way[16][33];
            uint8_t *comp_ptrs[16];
            for (int i = 0; i < 16; i++)
                comp_ptrs[i] = comp_16way[i];
            keygen_ge_to_pubkey_bytes_16way(ge_points, comp_ptrs, NULL);

            for (int i = 0; i < 16; i++) {
                uint8_t comp_scalar[33];
                keygen_ge_to_pubkey_bytes(&ge_points[i], comp_scalar, NULL);
                if (memcmp(comp_16way[i], comp_scalar, 33) != 0) {
                    printf("  [FAIL] 8.8d NULL uncompressed: lane%d compressed mismatch\n", i);
                    pass = 0;
                    fail_count++;
                }
            }
        }

        /* Test 3: partial lane NULL pointers (even lanes NULL, odd lanes valid) */
        {
            uint8_t comp_16way[16][33];
            uint8_t uncomp_16way[16][65];
            uint8_t *comp_ptrs[16], *uncomp_ptrs[16];
            for (int i = 0; i < 16; i++) {
                comp_ptrs[i]   = (i % 2 == 0) ? NULL : comp_16way[i];
                uncomp_ptrs[i] = (i % 2 == 0) ? uncomp_16way[i] : NULL;
            }
            keygen_ge_to_pubkey_bytes_16way(ge_points, comp_ptrs, uncomp_ptrs);

            for (int i = 0; i < 16; i++) {
                if (i % 2 == 0) {
                    /* Even lane: uncompressed should be correct */
                    uint8_t uncomp_scalar[65];
                    keygen_ge_to_pubkey_bytes(&ge_points[i], NULL, uncomp_scalar);
                    if (memcmp(uncomp_16way[i], uncomp_scalar, 65) != 0) {
                        printf("  [FAIL] 8.8d partial NULL: lane%d uncompressed mismatch\n", i);
                        pass = 0;
                        fail_count++;
                    }
                } else {
                    /* Odd lane: compressed should be correct */
                    uint8_t comp_scalar[33];
                    keygen_ge_to_pubkey_bytes(&ge_points[i], comp_scalar, NULL);
                    if (memcmp(comp_16way[i], comp_scalar, 33) != 0) {
                        printf("  [FAIL] 8.8d partial NULL: lane%d compressed mismatch\n", i);
                        pass = 0;
                        fail_count++;
                    }
                }
            }
        }

        if (pass) {
            printf("  [PASS] 8.8d NULL pointer skip: all combinations handled correctly\n");
            pass_count++;
        }
    }
}

/* ------------------------------------------------------------------
 * test_avx512ifma_16way_full_pipeline
 *
 * Verify end-to-end correctness of the entire 16-way concurrent pipeline under __AVX512IFMA__,
 * using the exact same interface chain as keysearch.c IFMA path:
 *   gej_add_ge_var_16way_soa  →  keygen_batch_normalize_rzr_16way
 *   →  hash160_16way_from_fe_soa  (direct SoA read, no fe_16x_load_ptrs)
 *
 * Final hash160 (compressed/uncompressed) compared with secp256k1 API lane by lane.
 *
 * Coverage scenarios:
 *   9.1  Single batch (BATCH_SIZE=16 steps): each of 16 chains advances 16 steps,
 *        each step uses gej_add_ge_var_16way_soa (step0 normed=0, others normed=1),
 *        keygen_batch_normalize_rzr_16way at end of batch (direct SoA output),
 *        hash160_16way_from_fe_soa for direct hash computation from SoA buffer,
 *        hash160 compared with API lane by lane, step by step.
 *   9.2  Multi-batch (3 batches): each batch with independent random base privkey,
 *        verify no state pollution between batches.
 * ------------------------------------------------------------------ */
static void test_avx512ifma_16way_full_pipeline(void) {
    printf("\n=== AVX512IFMA 16-way full pipeline end-to-end tests ===\n");

#define PIPE_STEPS 16   /* steps per batch */
#define PIPE_BATCH 3    /* number of batches */

    int all_pass = 1;

    /* Same as keysearch.c: gej_buf + rzr_buf + SoA output buffers + work buffer */
    secp256k1_gej gej_buf[16][PIPE_STEPS];
    secp256k1_fe  rzr_buf[16][PIPE_STEPS];
    secp256k1_fe_16x fe_x_soa_buf[PIPE_STEPS];
    secp256k1_fe_16x fe_y_soa_buf[PIPE_STEPS];
    secp256k1_ge *ge_work_buf = malloc(16 * PIPE_STEPS * sizeof(secp256k1_ge));
    if (!ge_work_buf) {
        printf("  [FAIL] ge_work_buf allocation failed\n");
        return;
    }

    secp256k1_scalar tweak;
    secp256k1_scalar_set_int(&tweak, 1);

    /* Initialize random context */
    rand_key_context rctx;
    if (rand_ctx_init(&rctx) != 0) {
        printf("  [FAIL] 9 rand_ctx_init failed\n");
        fail_count++;
        return;
    }

    for (int batch = 0; batch < PIPE_BATCH; batch++) {
        /* Initialize starting gej and scalar for 16 chains (random privkeys) */
        secp256k1_gej chain_gej[16];
        secp256k1_scalar chain_scalar[16];
        secp256k1_scalar init_scalar[16];  /* save initial scalar for API comparison */

        for (int ch = 0; ch < 16; ch++) {
            uint8_t pk[32];
            if (gen_random_key(pk, &rctx) != 0) {
                printf("  [FAIL] 9 batch%d ch%d gen_random_key failed\n", batch, ch);
                all_pass = 0;
                fail_count++;
                continue;
            }
            int overflow = 0;
            secp256k1_scalar_set_b32(&chain_scalar[ch], pk, &overflow);
            init_scalar[ch] = chain_scalar[ch];  /* save initial value */
            keygen_privkey_to_gej(secp_ctx, pk, &chain_gej[ch]);
        }

        /* Same as keysearch.c IFMA path: SoA persistent layout + rzr collection
         *
         * Storage convention (must match keygen_batch_normalize_rzr expectation):
         *   gej_buf[ch][step] = INPUT point before add (same as non-IFMA path)
         *   rzr_buf[ch][step] = Z(output) / Z(input) = Z(gej_buf[ch][step+1]) / Z(gej_buf[ch][step])
         */
        secp256k1_gej_16x chain_soa, next_soa;
        secp256k1_gej next_gej_16[16];  /* must persist across iterations for input storage */
        for (int step = 0; step < PIPE_STEPS; step++) {
            secp256k1_fe step_rzr[16];

            if (step == 0) {
                /* First step: AoS->SoA conversion, normed=0 */
                gej_16x_load(&chain_soa, chain_gej);
                /* Store input point BEFORE add (non-IFMA convention) */
                for (int ch = 0; ch < 16; ch++)
                    gej_buf[ch][0] = chain_gej[ch];
                gej_add_ge_var_16way_soa_compat(&next_soa, &chain_soa, &G_affine,
                                         next_gej_16, step_rzr, 0);
            } else {
                /* Store input point BEFORE add: previous output = current input */
                for (int ch = 0; ch < 16; ch++)
                    gej_buf[ch][step] = next_gej_16[ch];
                /* Subsequent steps: SoA input already available, normed=1 */
                gej_add_ge_var_16way_soa_compat(&next_soa, &chain_soa, &G_affine,
                                         next_gej_16, step_rzr, 1);
            }

            for (int ch = 0; ch < 16; ch++) {
                rzr_buf[ch][step] = step_rzr[ch];
                secp256k1_scalar_add(&chain_scalar[ch], &chain_scalar[ch], &tweak);
            }

            /* Persist SoA state for next iteration */
            chain_soa = next_soa;
        }

        /* Same as keysearch.c: use keygen_batch_normalize_rzr_16way for direct SoA output */
        int chain_valid_steps[16];
        for (int ch = 0; ch < 16; ch++)
            chain_valid_steps[ch] = PIPE_STEPS;
        keygen_batch_normalize_rzr_16way((const secp256k1_gej *)gej_buf,
                                         fe_x_soa_buf, fe_y_soa_buf,
                                         (const secp256k1_fe *)rzr_buf,
                                         ge_work_buf, chain_valid_steps, PIPE_STEPS);

        /* Verify hash160 step by step, lane by lane */
        for (int step = 0; step < PIPE_STEPS; step++) {
            /* Same as keysearch.c IFMA fast path:
             * Read SoA affine coordinates directly from keygen_batch_normalize_rzr_16way output
             * → hash160_16way_from_fe_soa (no fe_16x_load_ptrs gather needed) */
            uint8_t avx_h160_comp[16][20];
            uint8_t avx_h160_uncomp[16][20];
            hash160_16way_from_fe_soa(&fe_x_soa_buf[step], &fe_y_soa_buf[step],
                                     avx_h160_comp, avx_h160_uncomp);

            /* Compare with API lane by lane */
            for (int lane = 0; lane < 16; lane++) {
                /* Compute reference hash160 via API: privkey = init_scalar[lane] + step
                 * (gej_buf stores INPUT before add, so ge_buf[step] = (init_scalar+step)*G) */
                secp256k1_scalar ref_scalar = init_scalar[lane];
                for (int i = 0; i < step; i++)
                    secp256k1_scalar_add(&ref_scalar, &ref_scalar, &tweak);
                uint8_t ref_pk[32];
                secp256k1_scalar_get_b32(ref_pk, &ref_scalar);

                secp256k1_pubkey api_pubkey;
                secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, ref_pk);

                uint8_t api_comp[33], api_uncomp[65];
                size_t  len_c = 33, len_u = 65;
                secp256k1_ec_pubkey_serialize(secp_ctx, api_comp,   &len_c,
                                              &api_pubkey, SECP256K1_EC_COMPRESSED);
                secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len_u,
                                              &api_pubkey, SECP256K1_EC_UNCOMPRESSED);

                uint8_t ref_h160_comp[20], ref_h160_uncomp[20];
                pubkey_bytes_to_hash160(api_comp,   33, ref_h160_comp);
                pubkey_bytes_to_hash160(api_uncomp, 65, ref_h160_uncomp);

                /* Compare compressed hash160 */
                if (memcmp(avx_h160_comp[lane], ref_h160_comp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_comp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_comp,       20, ref_hex);
                    printf("  [FAIL] 9 batch%d step%d lane%d compressed hash160: "
                           "AVX512=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }

                /* Compare uncompressed hash160 */
                if (memcmp(avx_h160_uncomp[lane], ref_h160_uncomp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_uncomp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_uncomp,        20, ref_hex);
                    printf("  [FAIL] 9 batch%d step%d lane%d uncompressed hash160: "
                           "AVX512=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }
    }

    if (all_pass) {
        printf("  [PASS] 9 AVX512IFMA 16-way full pipeline (%d batches x %d steps x 16 lanes x compressed+uncompressed) all passed\n",
               PIPE_BATCH, PIPE_STEPS);
        pass_count++;
    }

    free(ge_work_buf);

#undef PIPE_STEPS
#undef PIPE_BATCH
}
#endif /* __AVX512IFMA__ && !USE_PUBKEY_API_ONLY */

/* ------------------------------------------------------------------
 * test_avx512f_16way_full_pipeline
 *
 * Verify end-to-end correctness of the entire 16-way concurrent pipeline under
 * !__AVX512IFMA__ && __AVX512F__, fully simulating the __AVX512F__ branch in keysearch.c:
 *
 *   keygen_privkey_to_gej  →  secp256k1_gej_add_ge_var (collect rzr)
 *   →  keygen_batch_normalize_rzr
 *   →  keygen_ge_to_pubkey_bytes  →  sha256_pad_block_33/sha256_pad_block2_65
 *   →  hash160_16way_compressed_prepadded
 *   →  hash160_16way_uncompressed_prepadded
 *
 * Final hash160 (compressed/uncompressed) compared with secp256k1 API lane by lane.
 *
 * Coverage scenarios:
 *   10.1  Single batch (BATCH_SIZE=64 steps): 1 chain advances 64 steps,
 *         grouped by 16, each group 16-way parallel hash160, compared with API lane by lane.
 *   10.2  Multi-batch (3 batches): each batch with independent random base privkey,
 *         verify no state pollution between batches.
 * ------------------------------------------------------------------ */
#if defined(__AVX512F__) && !defined(__AVX512IFMA__) && !defined(USE_PUBKEY_API_ONLY)
static void test_avx512f_16way_full_pipeline(void) {
    printf("\n=== AVX512F 16-way full pipeline end-to-end tests (!IFMA path) ===\n");

#define AVX512F_BATCH_STEPS 64   /* steps per batch, must be multiple of 16 */
#define AVX512F_BATCHES     3    /* number of batches */

    int all_pass = 1;

    secp256k1_scalar tweak;
    secp256k1_scalar_set_int(&tweak, 1);

    /* Initialize random context */
    rand_key_context rctx;
    if (rand_ctx_init(&rctx) != 0) {
        printf("  [FAIL] 10 rand_ctx_init failed\n");
        fail_count++;
        return;
    }

    for (int batch = 0; batch < AVX512F_BATCHES; batch++) {
        /* Generate random base privkey */
        uint8_t base_pk[32];
        if (gen_random_key(base_pk, &rctx) != 0) {
            printf("  [FAIL] 10 batch%d gen_random_key failed\n", batch);
            all_pass = 0;
            fail_count++;
            continue;
        }

        secp256k1_scalar base_scalar, cur_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_pk, &overflow);
        cur_scalar = base_scalar;

        /* Generate starting Jacobian point from base privkey */
        secp256k1_gej cur_gej;
        if (keygen_privkey_to_gej(secp_ctx, base_pk, &cur_gej) != 0) {
            printf("  [FAIL] 10 batch%d: keygen_privkey_to_gej failed\n", batch);
            all_pass = 0;
            fail_count++;
            continue;
        }

        /* Accumulate AVX512F_BATCH_STEPS Jacobian points, collect rzr simultaneously */
        secp256k1_gej gej_batch[AVX512F_BATCH_STEPS];
        secp256k1_ge  ge_batch[AVX512F_BATCH_STEPS];
        secp256k1_fe  rzr_batch[AVX512F_BATCH_STEPS];

        secp256k1_gej next_gej;
        for (int b = 0; b < AVX512F_BATCH_STEPS; b++) {
            gej_batch[b] = cur_gej;

            if (b < AVX512F_BATCH_STEPS - 1) {
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
                secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak);
            }
        }

        /* Batch normalize (using rzr incremental factor, fully consistent with __AVX512F__ branch) */
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch,
                                   (size_t)AVX512F_BATCH_STEPS);

        /* Group by 16, 16-way parallel hash160, compare with API lane by lane */
        for (int b = 0; b < AVX512F_BATCH_STEPS; b += 16) {
            int valid_count = AVX512F_BATCH_STEPS - b;
            if (valid_count > 16) valid_count = 16;

            uint8_t comp_bufs[16][64];
            uint8_t uncomp_bufs[16][128];
            const uint8_t *comp_ptrs[16];
            const uint8_t *uncomp_ptrs[16];

            for (int lane = 0; lane < 16; lane++) {
                /* Pad with last valid point when fewer than 16 (consistent with keysearch.c logic) */
                int idx = b + (lane < valid_count ? lane : valid_count - 1);
                if (ge_batch[idx].infinity) {
                    memset(comp_bufs[lane], 0, 64);
                    memset(uncomp_bufs[lane], 0, 128);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                } else {
                    keygen_ge_to_pubkey_bytes(&ge_batch[idx],
                                             comp_bufs[lane], uncomp_bufs[lane]);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                comp_ptrs[lane]   = comp_bufs[lane];
                uncomp_ptrs[lane] = uncomp_bufs[lane];
            }

            /* 16-way parallel hash160 */
            uint8_t avx_h160_comp[16][20];
            uint8_t avx_h160_uncomp[16][20];
            hash160_16way_compressed_prepadded(comp_ptrs,   avx_h160_comp);
            hash160_16way_uncompressed_prepadded(uncomp_ptrs, avx_h160_uncomp);

            /* Compare with API lane by lane (valid lanes only) */
            for (int lane = 0; lane < valid_count; lane++) {
                int step = b + lane;
                /* privkey = base + step (gej_batch[step] corresponds to base_scalar + step * 1) */
                secp256k1_scalar ref_scalar = base_scalar;
                for (int i = 0; i < step; i++)
                    secp256k1_scalar_add(&ref_scalar, &ref_scalar, &tweak);

                uint8_t ref_pk[32];
                secp256k1_scalar_get_b32(ref_pk, &ref_scalar);

                secp256k1_pubkey api_pubkey;
                secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, ref_pk);

                uint8_t api_comp[33], api_uncomp[65];
                size_t len_c = 33, len_u = 65;
                secp256k1_ec_pubkey_serialize(secp_ctx, api_comp,   &len_c,
                                              &api_pubkey, SECP256K1_EC_COMPRESSED);
                secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len_u,
                                              &api_pubkey, SECP256K1_EC_UNCOMPRESSED);

                uint8_t ref_h160_comp[20], ref_h160_uncomp[20];
                pubkey_bytes_to_hash160(api_comp,   33, ref_h160_comp);
                pubkey_bytes_to_hash160(api_uncomp, 65, ref_h160_uncomp);

                /* Compare compressed hash160 */
                if (memcmp(avx_h160_comp[lane], ref_h160_comp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_comp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_comp,       20, ref_hex);
                    printf("  [FAIL] 10 batch%d step%d lane%d compressed hash160: "
                           "AVX512F=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }

                /* Compare uncompressed hash160 */
                if (memcmp(avx_h160_uncomp[lane], ref_h160_uncomp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_uncomp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_uncomp,        20, ref_hex);
                    printf("  [FAIL] 10 batch%d step%d lane%d uncompressed hash160: "
                           "AVX512F=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }
    }

    if (all_pass) {
        printf("  [PASS] 10 AVX512F 16-way full pipeline (%d batches x %d steps x 16 lanes x compressed+uncompressed) all passed\n",
               AVX512F_BATCHES, AVX512F_BATCH_STEPS);
        pass_count++;
    }

#undef AVX512F_BATCH_STEPS
#undef AVX512F_BATCHES
}
#endif /* __AVX512F__ && !__AVX512IFMA__ && !USE_PUBKEY_API_ONLY */

/* ------------------------------------------------------------------
 * test_avx2_8way_full_pipeline
 *
 * Verify end-to-end correctness of the entire 8-way concurrent pipeline under
 * !__AVX512IFMA__ && !__AVX512F__ && __AVX2__, fully simulating the __AVX2__ branch
 * in keysearch.c:
 *
 *   keygen_privkey_to_gej  →  secp256k1_gej_add_ge_var (collect rzr)
 *   →  keygen_batch_normalize_rzr
 *   →  keygen_ge_to_pubkey_bytes  →  sha256_pad_block_33/sha256_pad_block2_65
 *   →  hash160_8way_compressed_prepadded
 *   →  hash160_8way_uncompressed_prepadded
 *
 * Final hash160 (compressed/uncompressed) compared with secp256k1 API lane by lane.
 *
 * Coverage scenarios:
 *   11.1  Single batch (BATCH_SIZE=64 steps): 1 chain advances 64 steps,
 *         grouped by 8, each group 8-way parallel hash160, compared with API lane by lane.
 *   11.2  Multi-batch (3 batches): each batch with independent random base privkey,
 *         verify no state pollution between batches.
 * ------------------------------------------------------------------ */
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(__AVX512IFMA__) && !defined(USE_PUBKEY_API_ONLY)
static void test_avx2_8way_full_pipeline(void) {
    printf("\n=== AVX2 8-way full pipeline end-to-end tests (!AVX512 path) ===\n");

#define AVX2_BATCH_STEPS 64   /* steps per batch, must be multiple of 8 */
#define AVX2_BATCHES     3    /* number of batches */

    int all_pass = 1;

    secp256k1_scalar tweak;
    secp256k1_scalar_set_int(&tweak, 1);

    /* Initialize random context */
    rand_key_context rctx;
    if (rand_ctx_init(&rctx) != 0) {
        printf("  [FAIL] 11 rand_ctx_init failed\n");
        fail_count++;
        return;
    }

    for (int batch = 0; batch < AVX2_BATCHES; batch++) {
        /* Generate random base privkey */
        uint8_t base_pk[32];
        if (gen_random_key(base_pk, &rctx) != 0) {
            printf("  [FAIL] 11 batch%d gen_random_key failed\n", batch);
            all_pass = 0;
            fail_count++;
            continue;
        }

        secp256k1_scalar base_scalar, cur_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_pk, &overflow);
        cur_scalar = base_scalar;

        /* Generate starting Jacobian point from base privkey */
        secp256k1_gej cur_gej;
        if (keygen_privkey_to_gej(secp_ctx, base_pk, &cur_gej) != 0) {
            printf("  [FAIL] 11 batch%d: keygen_privkey_to_gej failed\n", batch);
            all_pass = 0;
            fail_count++;
            continue;
        }

        /* Accumulate AVX2_BATCH_STEPS Jacobian points, collect rzr simultaneously */
        secp256k1_gej gej_batch[AVX2_BATCH_STEPS];
        secp256k1_ge  ge_batch[AVX2_BATCH_STEPS];
        secp256k1_fe  rzr_batch[AVX2_BATCH_STEPS];

        secp256k1_gej next_gej;
        for (int b = 0; b < AVX2_BATCH_STEPS; b++) {
            gej_batch[b] = cur_gej;

            if (b < AVX2_BATCH_STEPS - 1) {
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
                secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak);
            }
        }

        /* Batch normalize (using rzr incremental factor, fully consistent with __AVX2__ branch) */
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch,
                                   (size_t)AVX2_BATCH_STEPS);

        /* Group by 8, 8-way parallel hash160, compare with API lane by lane */
        for (int b = 0; b < AVX2_BATCH_STEPS; b += 8) {
            int valid_count = AVX2_BATCH_STEPS - b;
            if (valid_count > 8) valid_count = 8;

            uint8_t comp_bufs[8][64];
            uint8_t uncomp_bufs[8][128];
            const uint8_t *comp_ptrs[8];
            const uint8_t *uncomp_ptrs[8];

            for (int lane = 0; lane < 8; lane++) {
                /* Pad with last valid point when fewer than 8 (consistent with keysearch.c logic) */
                int idx = b + (lane < valid_count ? lane : valid_count - 1);
                if (ge_batch[idx].infinity) {
                    memset(comp_bufs[lane], 0, 64);
                    memset(uncomp_bufs[lane], 0, 128);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                } else {
                    keygen_ge_to_pubkey_bytes(&ge_batch[idx],
                                             comp_bufs[lane], uncomp_bufs[lane]);
                    sha256_pad_block_33(comp_bufs[lane]);
                    sha256_pad_block2_65(uncomp_bufs[lane]);
                }
                comp_ptrs[lane]   = comp_bufs[lane];
                uncomp_ptrs[lane] = uncomp_bufs[lane];
            }

            /* 8-way parallel hash160 */
            uint8_t avx_h160_comp[8][20];
            uint8_t avx_h160_uncomp[8][20];
            hash160_8way_compressed_prepadded(comp_ptrs,   avx_h160_comp);
            hash160_8way_uncompressed_prepadded(uncomp_ptrs, avx_h160_uncomp);

            /* Compare with API lane by lane (valid lanes only) */
            for (int lane = 0; lane < valid_count; lane++) {
                int step = b + lane;
                /* privkey = base + step */
                secp256k1_scalar ref_scalar = base_scalar;
                for (int i = 0; i < step; i++)
                    secp256k1_scalar_add(&ref_scalar, &ref_scalar, &tweak);

                uint8_t ref_pk[32];
                secp256k1_scalar_get_b32(ref_pk, &ref_scalar);

                secp256k1_pubkey api_pubkey;
                secp256k1_ec_pubkey_create(secp_ctx, &api_pubkey, ref_pk);

                uint8_t api_comp[33], api_uncomp[65];
                size_t len_c = 33, len_u = 65;
                secp256k1_ec_pubkey_serialize(secp_ctx, api_comp,   &len_c,
                                              &api_pubkey, SECP256K1_EC_COMPRESSED);
                secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len_u,
                                              &api_pubkey, SECP256K1_EC_UNCOMPRESSED);

                uint8_t ref_h160_comp[20], ref_h160_uncomp[20];
                pubkey_bytes_to_hash160(api_comp,   33, ref_h160_comp);
                pubkey_bytes_to_hash160(api_uncomp, 65, ref_h160_uncomp);

                /* Compare compressed hash160 */
                if (memcmp(avx_h160_comp[lane], ref_h160_comp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_comp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_comp,       20, ref_hex);
                    printf("  [FAIL] 11 batch%d step%d lane%d compressed hash160: "
                           "AVX2=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }

                /* Compare uncompressed hash160 */
                if (memcmp(avx_h160_uncomp[lane], ref_h160_uncomp, 20) != 0) {
                    char avx_hex[41], ref_hex[41];
                    bytes_to_hex_helper(avx_h160_uncomp[lane], 20, avx_hex);
                    bytes_to_hex_helper(ref_h160_uncomp,        20, ref_hex);
                    printf("  [FAIL] 11 batch%d step%d lane%d uncompressed hash160: "
                           "AVX2=%s API=%s\n",
                           batch, step, lane, avx_hex, ref_hex);
                    all_pass = 0;
                    fail_count++;
                }
            }
        }
    }

    if (all_pass) {
        printf("  [PASS] 11 AVX2 8-way full pipeline (%d batches x %d steps x 8 lanes x compressed+uncompressed) all passed\n",
               AVX2_BATCHES, AVX2_BATCH_STEPS);
        pass_count++;
    }

#undef AVX2_BATCH_STEPS
#undef AVX2_BATCHES
}
#endif /* __AVX2__ && !__AVX512F__ && !__AVX512IFMA__ && !USE_PUBKEY_API_ONLY */

#endif /* USE_PUBKEY_API_ONLY */

/* ===================== main ===================== */

int main(void) {
    printf("========================================\n");
    printf("  keysearch correctness verification program\n");
    printf("========================================\n");

    secp_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!secp_ctx) {
        fprintf(stderr, "Error: failed to initialize secp256k1\n");
        return 1;
    }

    test_sha256();
    test_ripemd160();
    test_hash160();
    test_incremental_pubkey();
#ifdef __AVX2__
    test_avx2_compress();
    test_hash160_8way();
#endif
#ifdef __AVX512F__
    if (__builtin_cpu_supports("avx512f")) {
        test_avx512_compress();
        test_hash160_16way();
    }
#endif
    test_ht_openaddr();
#ifdef __AVX2__
    test_ht_contains_8way_func();
#endif
#ifdef __AVX512F__
    if (__builtin_cpu_supports("avx512f")) {
        test_ht_contains_16way_func();
    }
#endif
    test_specialized_interfaces();
#ifndef USE_PUBKEY_API_ONLY
    /* Initialize global generator G (for use by test_keygen_internal / test_search_key_privkey_pubkey) */
    if (keygen_init_generator(secp_ctx, &G_affine) != 0) {
        fprintf(stderr, "Error: keygen_init_generator failed, skipping internal interface tests\n");
    } else {
        test_keygen_internal();
        test_search_key_privkey_pubkey();
#ifdef __AVX512IFMA__
        if (__builtin_cpu_supports("avx512ifma")) {
            test_gej_add_ge_var_16way();
            test_gej_add_ge_var_16way_soa();
            test_keygen_ge_to_pubkey_bytes_16way();
            test_avx512ifma_16way_full_pipeline();
        }
#endif
#if defined(__AVX512F__) && !defined(__AVX512IFMA__)
        if (__builtin_cpu_supports("avx512f")) {
            test_avx512f_16way_full_pipeline();
        }
#endif
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(__AVX512IFMA__)
        test_avx2_8way_full_pipeline();
#endif
    }
#endif

    secp256k1_context_destroy(secp_ctx);

    /* GPU algorithm CPU-side simulation tests */
    run_gpu_tests();

    printf("\n========================================\n");
    printf("  Passed: %d  Failed: %d\n", pass_count, fail_count);
    printf("========================================\n");

    return (fail_count > 0) ? 1 : 0;
}


