/*
 * hash_utils_avx512.c
 * AVX-512 specialized 16-way parallel hash160 function implementation
 */
#include "hash_utils.h"
#include <string.h>
#include <immintrin.h>

#ifdef __AVX512F__

/* Forward declaration of sha256_compress_avx512 */
void sha256_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16]);
/* Forward declaration of ripemd160_compress_avx512 */
void ripemd160_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16]);

/* SHA256 initial state constants */
static const uint32_t sha256_init_state[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* RIPEMD160 initial state constants */
static const uint32_t rmd160_init_state[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/* Construct SHA256 padded block (single block, message <= 55 bytes) */
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

/* Construct SHA256 first block (64-byte message block, no padding) */
static void make_sha256_block_raw(const uint8_t *msg64, uint8_t block[64])
{
    memcpy(block, msg64, 64);
}

/* Construct SHA256 second padded block (tail block for 65-byte message) */
static void make_sha256_block2_65(const uint8_t last_byte, uint8_t block[64])
{
    memset(block, 0, 64);
    block[0] = last_byte;
    block[1] = 0x80;
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

/* Convert SHA256 state[8] to 32-byte big-endian digest */
static void sha256_state_to_bytes(const uint32_t state[8], uint8_t out[32])
{
    for (int i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t)(state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(state[i]);
    }
}

/* Construct RIPEMD160 padded block (single block, message <= 55 bytes) */
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

/* Convert RIPEMD160 state[5] to 20-byte little-endian digest */
static void rmd160_state_to_bytes(const uint32_t state[5], uint8_t out[20])
{
    for (int i = 0; i < 5; i++) {
        out[i * 4 + 0] = (uint8_t)(state[i]);
        out[i * 4 + 1] = (uint8_t)(state[i] >> 8);
        out[i * 4 + 2] = (uint8_t)(state[i] >> 16);
        out[i * 4 + 3] = (uint8_t)(state[i] >> 24);
    }
}

/*
 * 16-way parallel hash160 (SHA256->RIPEMD160) for compressed public keys (33 bytes)
 */
__attribute__((target("avx512f")))
void hash160_16way_compressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20])
{
    uint8_t sha_blocks[16][64];
    uint32_t sha_states[16][8];
    uint32_t *sha_state_ptrs[16];
    const uint8_t *sha_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        make_sha256_block(pubkeys[i], 33, sha_blocks[i]);
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = sha_blocks[i];
    }

    sha256_compress_avx512(sha_state_ptrs, sha_block_ptrs);

    uint8_t sha_digests[16][32];
    for (int i = 0; i < 16; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
    }

    uint8_t rmd_blocks[16][64];
    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];
    const uint8_t *rmd_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        make_rmd160_block(sha_digests[i], 32, rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    ripemd160_compress_avx512(rmd_state_ptrs, rmd_block_ptrs);

    for (int i = 0; i < 16; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 16-way parallel hash160 for compressed public keys (pre-padded, zero-copy)
 * blocks[16]: array of 64-byte block pointers with SHA256 padding already applied in-place by caller
 */
__attribute__((target("avx512f")))
void hash160_16way_compressed_prepadded(const uint8_t *blocks[16], uint8_t hash160s[16][20])
{
    uint32_t sha_states[16][8];
    uint32_t *sha_state_ptrs[16];

    for (int i = 0; i < 16; i++) {
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
    }

    sha256_compress_avx512(sha_state_ptrs, blocks);

    /* sha_digests extended to 64 bytes, SHA256 digest written to first 32 bytes, last 32 bytes construct RIPEMD160 padding in-place */
    uint8_t sha_digests[16][64];
    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];
    const uint8_t *rmd_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
        /* Construct RIPEMD160 padded block in-place: 32-byte message, little-endian bit length 256 */
        sha_digests[i][32] = 0x80;
        memset(&sha_digests[i][33], 0, 23);
        /* Little-endian bit length 256 = 0x0000000000000100 */
        sha_digests[i][56] = 0x00; sha_digests[i][57] = 0x01;
        sha_digests[i][58] = 0x00; sha_digests[i][59] = 0x00;
        sha_digests[i][60] = 0x00; sha_digests[i][61] = 0x00;
        sha_digests[i][62] = 0x00; sha_digests[i][63] = 0x00;
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = sha_digests[i];
    }

    ripemd160_compress_avx512(rmd_state_ptrs, rmd_block_ptrs);

    for (int i = 0; i < 16; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 16-way parallel hash160 for uncompressed public keys (pre-padded, zero-copy)
 * bufs[16]: array of 128-byte buffer pointers with SHA256 padding already applied in-place by caller
 *           buf[0..63]  = SHA256 block1 (first 64 bytes of pubkey, no padding needed)
 *           buf[64..127]= SHA256 block2 (padding already applied)
 */
__attribute__((target("avx512f")))
void hash160_16way_uncompressed_prepadded(const uint8_t *bufs[16], uint8_t hash160s[16][20])
{
    uint32_t sha_states[16][8];
    uint32_t *sha_state_ptrs[16];
    const uint8_t *sha_block_ptrs[16];

    /* First pass: process block1 (buf[0..63]) */
    for (int i = 0; i < 16; i++) {
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = bufs[i];
    }

    sha256_compress_avx512(sha_state_ptrs, sha_block_ptrs);

    /* Second pass: process block2 (buf[64..127]) */
    for (int i = 0; i < 16; i++) {
        sha_block_ptrs[i] = bufs[i] + 64;
    }

    sha256_compress_avx512(sha_state_ptrs, sha_block_ptrs);

    /* sha_digests extended to 64 bytes, construct RIPEMD160 padding in-place */
    uint8_t sha_digests[16][64];
    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];
    const uint8_t *rmd_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
        /* Construct RIPEMD160 padded block in-place: 32-byte message, little-endian bit length 256 */
        sha_digests[i][32] = 0x80;
        memset(&sha_digests[i][33], 0, 23);
        /* Little-endian bit length 256 = 0x0000000000000100 */
        sha_digests[i][56] = 0x00; sha_digests[i][57] = 0x01;
        sha_digests[i][58] = 0x00; sha_digests[i][59] = 0x00;
        sha_digests[i][60] = 0x00; sha_digests[i][61] = 0x00;
        sha_digests[i][62] = 0x00; sha_digests[i][63] = 0x00;
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = sha_digests[i];
    }

    ripemd160_compress_avx512(rmd_state_ptrs, rmd_block_ptrs);

    for (int i = 0; i < 16; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 16-way parallel hash160 (SHA256->RIPEMD160) for uncompressed public keys (65 bytes)
 */
__attribute__((target("avx512f")))
void hash160_16way_uncompressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20])
{
    uint8_t sha_blocks1[16][64];
    uint32_t sha_states[16][8];
    uint32_t *sha_state_ptrs[16];
    const uint8_t *sha_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        make_sha256_block_raw(pubkeys[i], sha_blocks1[i]);
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = sha_blocks1[i];
    }

    sha256_compress_avx512(sha_state_ptrs, sha_block_ptrs);

    uint8_t sha_blocks2[16][64];
    for (int i = 0; i < 16; i++) {
        make_sha256_block2_65(pubkeys[i][64], sha_blocks2[i]);
        sha_block_ptrs[i] = sha_blocks2[i];
    }

    sha256_compress_avx512(sha_state_ptrs, sha_block_ptrs);

    uint8_t sha_digests[16][32];
    for (int i = 0; i < 16; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
    }

    uint8_t rmd_blocks[16][64];
    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];
    const uint8_t *rmd_block_ptrs[16];

    for (int i = 0; i < 16; i++) {
        make_rmd160_block(sha_digests[i], 32, rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    ripemd160_compress_avx512(rmd_state_ptrs, rmd_block_ptrs);

    for (int i = 0; i < 16; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 16-way parallel hash table lookup: search for 16 hash160 values simultaneously
 * Returns 16-bit hit mask (bit i set means lane i matched)
 */
__attribute__((target("avx512f")))
uint16_t ht_contains_16way(const uint8_t *h160s[16])
{
    uint32_t fps[16];
    uint32_t idxs[16];

    for (int i = 0; i < 16; i++) {
        const uint8_t *h = h160s[i];
        uint32_t fp = ((uint32_t)h[0] << 24) |
                      ((uint32_t)h[1] << 16) |
                      ((uint32_t)h[2] <<  8) |
                       (uint32_t)h[3];
        if (fp == 0)
            fp = 1;
        fps[i] = fp;

        idxs[i] = (fp * 2654435761u) & ht_mask;
    }

    uint16_t result = 0;

    for (int i = 0; i < 16; i++) {
        uint32_t idx = idxs[i];
        uint32_t fp  = fps[i];
        const uint8_t *h = h160s[i];

        while (1) {
            uint32_t slot_fp = ht_slots[idx].fp;
            if (slot_fp == 0)
                break;
            if (slot_fp == fp && memcmp(ht_slots[idx].h160, h, 20) == 0) {
                result |= (uint16_t)(1u << i);
                break;
            }
            idx = (idx + 1) & ht_mask;
        }
    }

    return result;
}

#endif /* __AVX512F__ */

