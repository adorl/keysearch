#include "hash_utils.h"
#include "sha256.h"
#include "ripemd160.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifndef USE_PUBKEY_API_ONLY
#include "secp256k1_keygen.h"
#else
#include <secp256k1.h>
#include "secp256k1_keygen.h"
#endif

/* secp256k1 context */
extern secp256k1_context *secp_ctx;

/* Base58 character table */
const char BASE58_CHARS[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* Base58 reverse lookup table: ASCII -> Base58 index (-1 means invalid character) */
static int base58_decode_map[256];
static int base58_decode_map_inited = 0;

/* Convert private key byte array to hex string */
void bytes_to_hex(const uint8_t *bytes, int len, char *hex_out)
{
    for (int i = 0; i < len; i++) {
        sprintf(hex_out + i * 2, "%02x", bytes[i]);
    }
    hex_out[len * 2] = '\0';
}

/* Double SHA256 hash */
void sha256d(const uint8_t *data, size_t len, uint8_t *out)
{
    uint8_t tmp[32];
    sha256(data, len, tmp);
    sha256(tmp, 32, out);
}

/* Base58Check encoding */
void base58check_encode(const uint8_t *payload, size_t payload_len, char *out)
{
    /* Compute checksum: first 4 bytes of sha256d */
    uint8_t checksum[32];
    sha256d(payload, payload_len, checksum);

    /* Concatenate payload + checksum[0..3] */
    uint8_t buf[payload_len + 4];
    memcpy(buf, payload, payload_len);
    memcpy(buf + payload_len, checksum, 4);
    size_t total = payload_len + 4;

    /* Count leading zero bytes */
    int leading_zeros = 0;
    for (size_t i = 0; i < total && buf[i] == 0; i++)
        leading_zeros++;

    /* Convert big number to Base58 */
    uint8_t digits[128] = {0};
    int digits_len = 0;

    for (size_t i = 0; i < total; i++) {
        int carry = buf[i];
        for (int j = 0; j < digits_len; j++) {
            carry += 256 * digits[j];
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry) {
            digits[digits_len++] = carry % 58;
            carry /= 58;
        }
    }

    /* Build output string (reverse order) */
    int pos = 0;
    for (int i = 0; i < leading_zeros; i++)
        out[pos++] = '1';
    for (int i = digits_len - 1; i >= 0; i--)
        out[pos++] = BASE58_CHARS[digits[i]];
    out[pos] = '\0';
}

/* Initialize reverse lookup table (executed only once) */
static void init_base58_decode_map(void)
{
    if (base58_decode_map_inited > 0)
        return;

    for (int i = 0; i < 256; i++)
        base58_decode_map[i] = -1;
    for (int i = 0; BASE58_CHARS[i] != '\0'; i++)
        base58_decode_map[(uint8_t)BASE58_CHARS[i]] = i;

    base58_decode_map_inited = 1;
}

/*
 * Decode Base58 string to byte array (big-endian)
 * Returns 0 on success, -1 on failure (invalid character or buffer overflow)
 */
static int base58_decode_bytes(const char *b58str, uint8_t *out, int *out_len)
{
    init_base58_decode_map();

    int cap = *out_len;
    int len = (int)strlen(b58str);

    /* Count leading '1' characters (corresponding to leading zero bytes) */
    int leading_ones = 0;
    while (leading_ones < len && b58str[leading_ones] == '1')
        leading_ones++;

    /* Use temporary buffer for big number arithmetic */
    uint8_t tmp[128] = {0};
    int tmp_len = 0;

    for (int i = leading_ones; i < len; i++) {
        int val = base58_decode_map[(uint8_t)b58str[i]];
        if (val < 0)
            return -1; /* invalid character */

        int carry = val;
        for (int j = tmp_len - 1; j >= 0; j--) {
            carry += 58 * tmp[j];
            tmp[j] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
        while (carry) {
            if (tmp_len >= 128)
                return -1; /* overflow */
            /* shift forward by one */
            memmove(tmp + 1, tmp, tmp_len);
            tmp[0] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
            tmp_len++;
        }
        if (tmp_len == 0 && val != 0)
            tmp_len = 1;
    }

    /* Compute total length */
    int total = leading_ones + tmp_len;
    if (total > cap)
        return -1; /* insufficient buffer */

    /* Write output */
    memset(out, 0, leading_ones);
    memcpy(out + leading_ones, tmp, tmp_len);
    *out_len = total;
    return 0;
}

/* Base58Check decoding: decode a Bitcoin address string back to a 20-byte RIPEMD160 hash */
int base58check_decode(const char *b58str, uint8_t *hash160_out)
{
    uint8_t buf[64];
    int buf_len = 64;

    /* Step 1: decode Base58 string to byte array */
    if (base58_decode_bytes(b58str, buf, &buf_len) != 0)
        return -1;

    /* Step 2: check length (version 1B + hash160 20B + checksum 4B = 25 bytes) */
    if (buf_len != 25)
        return -1;

    /* Step 3: verify checksum */
    uint8_t hash[32];
    sha256d(buf, 21, hash); /* double SHA256 over first 21 bytes */
    if (memcmp(hash, buf + 21, 4) != 0)
        return -2;

    /* Step 4: re-encode to base58 and compare with original */
    char address[64];
    base58check_encode(buf, 21, address);
    if (strcmp(b58str, address) != 0)
        return -2;

    /* Step 5: extract hash160 (skip version byte) */
    if (hash160_out != NULL)
        memcpy(hash160_out, buf + 1, 20);

    return 0;
}

/*
 * Compute hash160 (SHA256 -> RIPEMD160) directly from serialized public key bytes
 */
void pubkey_bytes_to_hash160(const uint8_t *pubkey_bytes, size_t len,
                            uint8_t *hash160_out)
{
    uint8_t sha256_result[32];
    if (len == 33) {
        sha256_33(pubkey_bytes, sha256_result);
    } else if (len == 65) {
        sha256_65(pubkey_bytes, sha256_result);
    } else {
        sha256(pubkey_bytes, len, sha256_result);
    }
    ripemd160_32(sha256_result, hash160_out);
}

/*
 * Compute hash160 (SHA256 -> RIPEMD160) for both compressed and uncompressed public keys
 * from a 32-byte private key.
 * compressed_hash160   : output for compressed pubkey hash160 (20 bytes), pass NULL to skip
 * uncompressed_hash160 : output for uncompressed pubkey hash160 (20 bytes), pass NULL to skip
 * Return value: 0 on success, -1 on failure (pubkey generation failed)
 */
int privkey_to_hash160(const uint8_t *privkey,
                       uint8_t *compressed_hash160,
                       uint8_t *uncompressed_hash160)
{
    uint8_t sha256_result[32];

    /* One elliptic curve operation to get the public key object */
    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey))
        return -1;

    /* ---- Compressed pubkey hash160 ---- */
    if (compressed_hash160 != NULL) {
        uint8_t pubkey_compressed[33];
        size_t pubkey_len1 = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_compressed, &pubkey_len1,
            &pubkey, SECP256K1_EC_COMPRESSED);
        sha256(pubkey_compressed, 33, sha256_result);
        ripemd160(sha256_result, 32, compressed_hash160);
    }

    /* ---- Uncompressed pubkey hash160 ---- */
    if (uncompressed_hash160 != NULL) {
        uint8_t pubkey_uncompressed[65];
        size_t pubkey_len2 = 65;
        secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_uncompressed, &pubkey_len2,
            &pubkey, SECP256K1_EC_UNCOMPRESSED);
        sha256(pubkey_uncompressed, 65, sha256_result);
        ripemd160(sha256_result, 32, uncompressed_hash160);
    }

    return 0;
}

/*
 * Compute both compressed and uncompressed Bitcoin addresses (P2PKH)
 * from a 32-byte private key simultaneously.
 */
int privkey_to_address(const uint8_t *privkey,
                       char *compressed_out,
                       char *uncompressed_out)
{
    uint8_t hash160_compressed[20];
    uint8_t hash160_uncompressed[20];
    uint8_t versioned[21];

    /* Reuse privkey_to_hash160 to compute both hash160 values */
    if (privkey_to_hash160(privkey, hash160_compressed, hash160_uncompressed) != 0)
        return -1;

    /* ---- Compressed address ---- */
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160_compressed, 20);
    base58check_encode(versioned, 21, compressed_out);

    /* ---- Uncompressed address ---- */
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160_uncompressed, 20);
    base58check_encode(versioned, 21, uncompressed_out);

    return 0;
}

/*
 * Compute P2WPKH (bech32) address from a private key
 * Uses compressed public key's hash160 as the witness program
 */
int privkey_to_bech32_p2wpkh(const uint8_t *privkey, char *bech32_out)
{
    uint8_t hash160_compressed[20];

    /* Compute hash160 of compressed public key */
    if (privkey_to_hash160(privkey, hash160_compressed, NULL) != 0)
        return -1;

    /* Encode as bech32 address: witness version 0, 20-byte witness program */
    if (bech32_encode_witness("bc", 0, hash160_compressed, 20, bech32_out) != 0)
        return -1;

    return 0;
}

struct ht_slot *ht_slots = NULL;
uint32_t ht_mask = 0;

/* Initialize hash table: allocate capacity slots (capacity must be a power of 2) */
int ht_init(uint32_t capacity)
{
    ht_slots = (struct ht_slot *)calloc(capacity, sizeof(struct ht_slot));
    if (!ht_slots)
        return -1;
    ht_mask = capacity - 1;
    return 0;
}

/* Free hash table memory */
void ht_free(void)
{
    free(ht_slots);
    ht_slots = NULL;
    ht_mask = 0;
}

/*
 * Knuth multiplicative hash: one multiplication on the 4-byte fingerprint fp
 * Returns slot index (already & ht_mask)
 */
static inline uint32_t ht_hash(uint32_t fp)
{
    return (fp * 2654435761u) & ht_mask;
}

/* Insert hash160 into hash table (linear probing) */
void ht_insert(const uint8_t *h160)
{
    /* Extract 4-byte fingerprint (big-endian) */
    uint32_t fp = ((uint32_t)h160[0] << 24) |
                  ((uint32_t)h160[1] << 16) |
                  ((uint32_t)h160[2] <<  8) |
                   (uint32_t)h160[3];
    /* Replace fp == 0 with 1 to avoid conflict with empty slot marker */
    if (fp == 0)
        fp = 1;

    uint32_t idx = ht_hash(fp);
    while (ht_slots[idx].fp != 0) {
        idx = (idx + 1) & ht_mask;
    }
    ht_slots[idx].fp = fp;
    memcpy(ht_slots[idx].h160, h160, 20);
}

/* Look up hash160 in hash table (linear probing) */
int ht_contains(const uint8_t *h160)
{
    uint32_t fp = ((uint32_t)h160[0] << 24) |
                  ((uint32_t)h160[1] << 16) |
                  ((uint32_t)h160[2] <<  8) |
                   (uint32_t)h160[3];
    if (fp == 0)
        fp = 1;

    uint32_t idx = ht_hash(fp);
    while (1) {
        uint32_t slot_fp = ht_slots[idx].fp;
        if (slot_fp == 0)
            return 0; /* empty slot, miss */
        if (slot_fp == fp && memcmp(ht_slots[idx].h160, h160, 20) == 0)
            return 1; /* hit */
        idx = (idx + 1) & ht_mask;
    }
}

#ifdef __AVX2__

/* SHA256 initial state constants */
static const uint32_t sha256_init_state[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* RIPEMD160 initial state constants */
static const uint32_t rmd160_init_state[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/*
 * Construct SHA256 padded block (single block, message <= 55 bytes)
 * message + 0x80 + zero padding + 8-byte big-endian message bit length
 */
static void make_sha256_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen >> 56);
    block[57] = (uint8_t)(bitlen >> 48);
    block[58] = (uint8_t)(bitlen >> 40);
    block[59] = (uint8_t)(bitlen >> 32);
    block[60] = (uint8_t)(bitlen >> 24);
    block[61] = (uint8_t)(bitlen >> 16);
    block[62] = (uint8_t)(bitlen >> 8);
    block[63] = (uint8_t)(bitlen);
}

/*
 * Construct SHA256 first block (64-byte message block, no padding)
 * Used for the first block of a 65-byte message (first 64 bytes)
 */
static void make_sha256_block_raw(const uint8_t *msg64, uint8_t block[64])
{
    memcpy(block, msg64, 64);
}

/*
 * Construct SHA256 second padded block (tail block for 65-byte message)
 * remaining 1 byte + 0x80 + zero padding + 8-byte big-endian bit length (65*8=520 bits)
 */
static void make_sha256_block2_65(const uint8_t last_byte, uint8_t block[64])
{
    memset(block, 0, 64);
    block[0] = last_byte;
    block[1] = 0x80;
    /* message bit length = 65 * 8 = 520 = 0x208 */
    uint64_t bitlen = 65ULL * 8;
    block[56] = (uint8_t)(bitlen >> 56);
    block[57] = (uint8_t)(bitlen >> 48);
    block[58] = (uint8_t)(bitlen >> 40);
    block[59] = (uint8_t)(bitlen >> 32);
    block[60] = (uint8_t)(bitlen >> 24);
    block[61] = (uint8_t)(bitlen >> 16);
    block[62] = (uint8_t)(bitlen >> 8);
    block[63] = (uint8_t)(bitlen);
}

/*
 * Convert SHA256 state[8] to 32-byte big-endian digest
 */
static void sha256_state_to_bytes(const uint32_t state[8], uint8_t out[32])
{
    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t)(state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(state[i]);
    }
}

/*
 * Construct RIPEMD160 padded block (single block, message <= 55 bytes)
 * message + 0x80 + zero padding + 8-byte little-endian message bit length
 */
static void make_rmd160_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen);
    block[57] = (uint8_t)(bitlen >> 8);
    block[58] = (uint8_t)(bitlen >> 16);
    block[59] = (uint8_t)(bitlen >> 24);
    block[60] = (uint8_t)(bitlen >> 32);
    block[61] = (uint8_t)(bitlen >> 40);
    block[62] = (uint8_t)(bitlen >> 48);
    block[63] = (uint8_t)(bitlen >> 56);
}

/*
 * Convert RIPEMD160 state[5] to 20-byte little-endian digest
 */
static void rmd160_state_to_bytes(const uint32_t state[5], uint8_t out[20])
{
    memcpy(out, state, 20);
}

/*
 * Construct RIPEMD160 padded block directly from SHA256 state (8 big-endian uint32_t)
 * SHA256 outputs 32 bytes (big-endian), used as RIPEMD160 message input
 */
static void make_rmd160_block_from_sha256_state(const uint32_t sha_state[8],
                                                uint8_t block[64])
{
    /* Write SHA256 state in big-endian order to first 32 bytes of block */
    for (int i = 0; i < 8; i++) {
        block[i * 4 + 0] = (uint8_t)(sha_state[i] >> 24);
        block[i * 4 + 1] = (uint8_t)(sha_state[i] >> 16);
        block[i * 4 + 2] = (uint8_t)(sha_state[i] >> 8);
        block[i * 4 + 3] = (uint8_t)(sha_state[i]);
    }
    /* RIPEMD160 padding: 0x80 + zero padding + 8-byte little-endian bit length (32*8=256 bits) */
    block[32] = 0x80;
    memset(block + 33, 0, 64 - 33 - 8);
    /* bit length 256 = 0x100, written in little-endian to last 8 bytes */
    block[56] = 0x00;
    block[57] = 0x01;
    block[58] = 0x00;
    block[59] = 0x00;
    block[60] = 0x00;
    block[61] = 0x00;
    block[62] = 0x00;
    block[63] = 0x00;
}

/*
 * 8-way parallel hash160 (SHA256->RIPEMD160) for compressed public keys (33 bytes)
 *
 * Compressed pubkey 33 bytes < 56 bytes, single-block SHA256 suffices
 * SHA256 output 32 bytes < 56 bytes, single-block RIPEMD160 suffices
 */
__attribute__((target("avx2")))
void hash160_8way_compressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20])
{
    /* ---- Step 1: construct 8-way SHA256 padded blocks (33-byte message) ---- */
    uint8_t sha_blocks[8][64];
    uint32_t sha_states[8][8];
    uint32_t *sha_state_ptrs[8];
    const uint8_t *sha_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_sha256_block(pubkeys[i], 33, sha_blocks[i]);
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = sha_blocks[i];
    }

    /* ---- Step 2: 8-way parallel SHA256 compression ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- Step 3: construct 8-way RIPEMD160 padded blocks directly from SHA256 state ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block_from_sha256_state(sha_states[i], rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- Step 4: 8-way parallel RIPEMD160 compression ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- Step 5: extract 8-way RIPEMD160 digests (20-byte little-endian) ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8-way parallel hash160 for compressed public keys (pre-padded, zero-copy)
 * Caller must write the 33-byte pubkey into buffer[0..32] of a 64-byte buffer,
 * and call sha256_pad_block_33() to complete SHA256 padding in-place
 */
__attribute__((target("avx2")))
void hash160_8way_compressed_prepadded(const uint8_t *blocks[8], uint8_t hash160s[8][20])
{
    /* ---- Step 1: use caller's pre-padded block directly, no copy needed ---- */
    uint32_t sha_states[8][8];
    uint32_t *sha_state_ptrs[8];

    for (int i = 0; i < 8; i++) {
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
    }

    /* ---- Step 2: 8-way parallel SHA256 compression ---- */
    sha256_compress_avx2(sha_state_ptrs, (const uint8_t **)blocks);

    /* ---- Step 3: construct 8-way RIPEMD160 padded blocks directly from SHA256 state ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block_from_sha256_state(sha_states[i], rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- Step 4: 8-way parallel RIPEMD160 compression ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- Step 5: extract 8-way RIPEMD160 digests (20-byte little-endian) ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8-way parallel hash160 for uncompressed public keys (pre-padded, zero-copy)
 *
 * Caller must write the 65-byte pubkey into buffer[0..64] of a 128-byte buffer,
 * and call sha256_pad_block2_65() to construct block2 in-place at buffer[64..127],
 * then pass the 128-byte buffer pointer to this function, avoiding internal make_sha256_block2_65
 */
__attribute__((target("avx2")))
void hash160_8way_uncompressed_prepadded(const uint8_t *bufs[8], uint8_t hash160s[8][20])
{
    /* ---- Step 1: use bufs[i] pointer directly as SHA256 first block (zero-copy) ---- */
    uint32_t sha_states[8][8];
    uint32_t *sha_state_ptrs[8];
    const uint8_t *sha_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = bufs[i];           /* block1 = buf[0..63] */
    }

    /* ---- Step 2: 8-way parallel SHA256 first compression ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- Step 3: use bufs[i]+64 pointer directly as SHA256 second block (zero-copy) ---- */
    for (int i = 0; i < 8; i++) {
        sha_block_ptrs[i] = bufs[i] + 64;      /* block2 = buf[64..127] */
    }

    /* ---- Step 4: 8-way parallel SHA256 second compression ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- Step 5: construct 8-way RIPEMD160 padded blocks directly from SHA256 state ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block_from_sha256_state(sha_states[i], rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- Step 6: 8-way parallel RIPEMD160 compression ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- Step 7: extract 8-way RIPEMD160 digests (20-byte little-endian) ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8-way parallel hash160 (SHA256 -> RIPEMD160) for uncompressed public keys (65 bytes)
 *
 * Uncompressed pubkey 65 bytes requires 2 SHA256 blocks:
 *   block1: first 64 bytes (raw data, no padding)
 *   block2: 65th byte + 0x80 + zero padding + 8-byte big-endian bit length (520 bits)
 */
__attribute__((target("avx2")))
void hash160_8way_uncompressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20])
{
    /* ---- Step 1: use pubkeys[i] raw pointer directly as SHA256 first block ---- */
    uint32_t sha_states[8][8];
    uint32_t *sha_state_ptrs[8];
    const uint8_t *sha_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = pubkeys[i];
    }

    /* ---- Step 2: 8-way parallel SHA256 first compression ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- Step 3: construct 8-way SHA256 second padded block (65th byte + padding) ---- */
    uint8_t sha_blocks2[8][64];
    for (int i = 0; i < 8; i++) {
        make_sha256_block2_65(pubkeys[i][64], sha_blocks2[i]);
        sha_block_ptrs[i] = sha_blocks2[i];
    }

    /* ---- Step 4: 8-way parallel SHA256 second compression (state already updated, continue accumulating) ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- Step 5: construct 8-way RIPEMD160 padded blocks directly from SHA256 state ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block_from_sha256_state(sha_states[i], rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- Step 6: 8-way parallel RIPEMD160 compression ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- Step 7: extract 8-way RIPEMD160 digests (20-byte little-endian) ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8-way parallel hash table lookup: search for 8 hash160 values simultaneously
 *
 * Strategy:
 *   1. Use scalar Knuth hash to compute 8-way slot indices (identical to ht_contains)
 *   2. Use AVX2 _mm256_set_epi32 to load 8-way fingerprints, _mm256_cmpeq_epi32 to compare simultaneously
 *   3. For fingerprint-matched lanes, perform scalar linear probing + memcmp 20-byte confirmation
 *   4. Return 8-bit hit mask (bit i set means lane i matched)
 */
__attribute__((target("avx2")))
uint8_t ht_contains_8way(const uint8_t *h160s[8])
{
    /* ---- Step 1: compute 8-way fingerprints and initial slot indices ---- */
    uint32_t fps[8];
    uint32_t idxs[8];

    for (int i = 0; i < 8; i++) {
        const uint8_t *h = h160s[i];
        uint32_t fp = ((uint32_t)h[0] << 24) |
                      ((uint32_t)h[1] << 16) |
                      ((uint32_t)h[2] <<  8) |
                       (uint32_t)h[3];
        if (fp == 0)
            fp = 1;
        fps[i] = fp;

        /* Knuth multiplicative hash (identical to ht_hash) */
        idxs[i] = (fp * 2654435761u) & ht_mask;
    }

    /* ---- Step 2: AVX2 load 8-way slot fingerprints and compare simultaneously ---- */

    /* Load fingerprints of 8 current slots */
    __m256i vslot = _mm256_set_epi32(
        (int32_t)ht_slots[idxs[7]].fp, (int32_t)ht_slots[idxs[6]].fp,
        (int32_t)ht_slots[idxs[5]].fp, (int32_t)ht_slots[idxs[4]].fp,
        (int32_t)ht_slots[idxs[3]].fp, (int32_t)ht_slots[idxs[2]].fp,
        (int32_t)ht_slots[idxs[1]].fp, (int32_t)ht_slots[idxs[0]].fp);

    /* Detect empty slots (fp == 0 means empty slot, direct miss) */
    __m256i vzero = _mm256_setzero_si256();
    __m256i vempty = _mm256_cmpeq_epi32(vslot, vzero); /* empty slot mask */

    int empty_mask = _mm256_movemask_epi8(vempty);

    /* ---- Step 3: for fingerprint-matched lanes, perform scalar linear probing + memcmp confirmation ---- */
    uint8_t result = 0;

    for (int i = 0; i < 8; i++) {
        /* Each lane corresponds to 4 consecutive bits in movemask (epi32 -> 4 bytes) */
        int lane_bit = 1 << (i * 4);

        if (empty_mask & lane_bit) {
            /* initial slot is empty, direct miss */
            continue;
        }

        /* Scalar linear probing: start from initial slot until match or empty slot found */
        uint32_t idx = idxs[i];
        uint32_t fp  = fps[i];
        const uint8_t *h = h160s[i];

        while (1) {
            uint32_t slot_fp = ht_slots[idx].fp;
            if (slot_fp == 0)
                break; /* empty slot, miss */
            if (slot_fp == fp && memcmp(ht_slots[idx].h160, h, 20) == 0) {
                result |= (uint8_t)(1 << i);
                break;
            }
            idx = (idx + 1) & ht_mask;
        }
    }

    return result;
}

#endif /* __AVX2__ */


