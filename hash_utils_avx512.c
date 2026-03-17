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
void sha256_compress_avx512_contig(uint32_t *states[16], const uint8_t *base, int stride);
/* Forward declaration of ripemd160_compress_avx512 */
void ripemd160_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16]);
void ripemd160_compress_avx512_contig(uint32_t *states[16], const uint8_t *base, int stride);

/* SHA256 initial state pre-expanded to 16 lanes to avoid per-loop copying */
static const uint32_t sha256_init_state_16way[16][8] = {
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
};

/* RIPEMD160 initial state pre-expanded to 16 lanes to avoid per-loop copying */
static const uint32_t rmd160_init_state_16way[16][5] = {
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
    {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0},
};

/*
 * Pre-filled RIPEMD160 blocks template for 32-byte messages (SHA256 digests).
 * Each block[16][64] has bytes [32..63] pre-filled with RIPEMD160 padding:
 *   block[32]    = 0x80 (padding marker)
 *   block[33..55]= 0x00 (23 zero bytes)
 *   block[56..63]= LE64(256) (bit length = 32*8 = 256)
 * Only the first 32 bytes need to be written per invocation.
 */
#define RMD_BLK_TEMPLATE_ROW    \
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x01, 0, 0, 0, 0, 0, 0 }
static const uint8_t rmd_blocks_template_16way[16][64] = {
    RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW,
    RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW,
    RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW,
    RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW, RMD_BLK_TEMPLATE_ROW
};
#undef RMD_BLK_TEMPLATE_ROW

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

/* Convert SHA256 state[8] to 32-byte big-endian digest (scalar, single lane) */
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
 * sha256_state_to_bytes_16way: vectorized big-endian byte-swap for 16 SHA256 states.
 * For each of the 8 state words, gathers 16 uint32_t values into a __m512i,
 * performs in-register byte-swap via _mm512_shuffle_epi8, then scatters
 * the 4-byte results into each lane's output buffer.
 * Replaces 16 x 8 = 128 scalar shift operations with 8 SIMD iterations.
 */
static void sha256_state_to_bytes_16way(uint32_t sha_states[16][8],
                                        uint8_t rmd_blocks[16][64])
{
    /*
     * AVX-512F only (no BW needed): byte-swap uint32_t using shift+mask+or.
     * For each 32-bit element ABCD -> DCBA:
     *   byte3 = (v >> 24) & 0xFF
     *   byte2 = (v >> 8)  & 0xFF00
     *   byte1 = (v << 8)  & 0xFF0000
     *   byte0 = (v << 24) & 0xFF000000
     */
    const __m512i mask_ff   = _mm512_set1_epi32(0x000000FF);
    const __m512i mask_ff00 = _mm512_set1_epi32(0x0000FF00);

    for (int w = 0; w < 8; w++) {
        /* Gather state word w from all 16 lanes into two __m512i (16 x uint32_t each) */
        uint32_t lo_buf[16] __attribute__((aligned(64)));
        uint32_t hi_buf[16] __attribute__((aligned(64)));
        for (int i = 0; i < 8; i++) {
            lo_buf[i] = sha_states[i][w];
            hi_buf[i] = sha_states[i + 8][w];
        }
        __m512i lo = _mm512_load_si512((const void *)lo_buf);
        __m512i hi = _mm512_load_si512((const void *)hi_buf);

        /* In-register 32-bit byte-swap using AVX-512F shift/mask/or */
        __m512i lo_b3 = _mm512_and_si512(_mm512_srli_epi32(lo, 24), mask_ff);
        __m512i lo_b2 = _mm512_and_si512(_mm512_srli_epi32(lo, 8),  mask_ff00);
        __m512i lo_b1 = _mm512_and_si512(_mm512_slli_epi32(lo, 8),  _mm512_set1_epi32(0x00FF0000));
        __m512i lo_b0 = _mm512_slli_epi32(lo, 24);
        lo = _mm512_or_si512(_mm512_or_si512(lo_b0, lo_b1), _mm512_or_si512(lo_b2, lo_b3));

        __m512i hi_b3 = _mm512_and_si512(_mm512_srli_epi32(hi, 24), mask_ff);
        __m512i hi_b2 = _mm512_and_si512(_mm512_srli_epi32(hi, 8),  mask_ff00);
        __m512i hi_b1 = _mm512_and_si512(_mm512_slli_epi32(hi, 8),  _mm512_set1_epi32(0x00FF0000));
        __m512i hi_b0 = _mm512_slli_epi32(hi, 24);
        hi = _mm512_or_si512(_mm512_or_si512(hi_b0, hi_b1), _mm512_or_si512(hi_b2, hi_b3));

        /* Store swapped 4 bytes to each lane's rmd_block at offset w*4 */
        _mm512_store_si512((__m512i *)lo_buf, lo);
        _mm512_store_si512((__m512i *)hi_buf, hi);

        int off = w * 4;
        for (int i = 0; i < 8; i++) {
            memcpy(&rmd_blocks[i][off], &lo_buf[i], 4);
        }
        for (int i = 0; i < 8; i++) {
            memcpy(&rmd_blocks[i + 8][off], &hi_buf[i], 4);
        }
    }
}

/* Convert RIPEMD160 state[5] to 20-byte little-endian digest.
 * On x86 (native little-endian), uint32_t memory layout already matches
 * the desired byte order, so a direct memcpy suffices. */
static void rmd160_state_to_bytes(const uint32_t state[5], uint8_t out[20])
{
    memcpy(out, state, 20);
}

static void hash160_16way_sha_init(uint32_t sha_states[16][8], uint32_t *sha_state_ptrs[16])
{
    /* Single memcpy(512 bytes) replaces 16x memcpy(32 bytes), more vectorization-friendly */
    memcpy(sha_states, sha256_init_state_16way, sizeof(sha256_init_state_16way));
    for (int i = 0; i < 16; i++) {
        sha_state_ptrs[i] = sha_states[i];
    }
}

static void hash160_16way_finalize_from_sha(uint32_t sha_states[16][8], uint8_t hash160s[16][20])
{
    /* Copy template with pre-filled padding; only first 32 bytes per row need updating */
    uint8_t rmd_blocks[16][64];
    memcpy(rmd_blocks, rmd_blocks_template_16way, sizeof(rmd_blocks_template_16way));

    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];
    const uint8_t *rmd_block_ptrs[16];

    memcpy(rmd_states, rmd160_init_state_16way, sizeof(rmd160_init_state_16way));
    /* Vectorized byte-swap: convert 16 SHA256 states to big-endian bytes in rmd_blocks */
    sha256_state_to_bytes_16way(sha_states, rmd_blocks);
    for (int i = 0; i < 16; i++) {
        rmd_state_ptrs[i] = rmd_states[i];
    }

    /* rmd_blocks[16][64] is contiguous, use gather-optimized variant */
    ripemd160_compress_avx512_contig(rmd_state_ptrs, (const uint8_t *)rmd_blocks, 64);

    for (int i = 0; i < 16; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

static void hash160_16way_prepadded_sha(const uint8_t *blocks1[16],
                                        const uint8_t *blocks2[16],
                                        uint8_t hash160s[16][20])
{
    uint32_t sha_states[16][8];
    uint32_t *sha_state_ptrs[16];

    hash160_16way_sha_init(sha_states, sha_state_ptrs);
    sha256_compress_avx512(sha_state_ptrs, blocks1);

    if (blocks2 != NULL) {
        sha256_compress_avx512(sha_state_ptrs, blocks2);
    }

    hash160_16way_finalize_from_sha(sha_states, hash160s);
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

    memcpy(sha_states, sha256_init_state_16way, sizeof(sha256_init_state_16way));
    for (int i = 0; i < 16; i++) {
        make_sha256_block(pubkeys[i], 33, sha_blocks[i]);
        sha_state_ptrs[i] = sha_states[i];
    }

    /* sha_blocks[16][64] is contiguous, use gather-optimized variant */
    sha256_compress_avx512_contig(sha_state_ptrs, (const uint8_t *)sha_blocks, 64);

    /* Reuse pre-filled template: only write first 32 bytes per block */
    uint8_t rmd_blocks[16][64];
    memcpy(rmd_blocks, rmd_blocks_template_16way, sizeof(rmd_blocks_template_16way));

    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];

    memcpy(rmd_states, rmd160_init_state_16way, sizeof(rmd160_init_state_16way));
    /* Vectorized byte-swap: convert 16 SHA256 states to big-endian bytes in rmd_blocks */
    sha256_state_to_bytes_16way(sha_states, rmd_blocks);
    for (int i = 0; i < 16; i++) {
        rmd_state_ptrs[i] = rmd_states[i];
    }

    /* rmd_blocks[16][64] is contiguous, use gather-optimized variant */
    ripemd160_compress_avx512_contig(rmd_state_ptrs, (const uint8_t *)rmd_blocks, 64);

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
    hash160_16way_prepadded_sha(blocks, NULL, hash160s);
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
    const uint8_t *blocks2[16];

    for (int i = 0; i < 16; i++) {
        blocks2[i] = bufs[i] + 64;
    }

    hash160_16way_prepadded_sha(bufs, blocks2, hash160s);
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

    memcpy(sha_states, sha256_init_state_16way, sizeof(sha256_init_state_16way));
    for (int i = 0; i < 16; i++) {
        make_sha256_block_raw(pubkeys[i], sha_blocks1[i]);
        sha_state_ptrs[i] = sha_states[i];
    }

    /* sha_blocks1[16][64] is contiguous, use gather-optimized variant */
    sha256_compress_avx512_contig(sha_state_ptrs, (const uint8_t *)sha_blocks1, 64);

    uint8_t sha_blocks2[16][64];
    for (int i = 0; i < 16; i++) {
        make_sha256_block2_65(pubkeys[i][64], sha_blocks2[i]);
    }

    /* sha_blocks2[16][64] is contiguous, use gather-optimized variant */
    sha256_compress_avx512_contig(sha_state_ptrs, (const uint8_t *)sha_blocks2, 64);

    /* Reuse pre-filled template: only write first 32 bytes per block */
    uint8_t rmd_blocks[16][64];
    memcpy(rmd_blocks, rmd_blocks_template_16way, sizeof(rmd_blocks_template_16way));

    uint32_t rmd_states[16][5];
    uint32_t *rmd_state_ptrs[16];

    memcpy(rmd_states, rmd160_init_state_16way, sizeof(rmd160_init_state_16way));
    /* Vectorized byte-swap: convert 16 SHA256 states to big-endian bytes in rmd_blocks */
    sha256_state_to_bytes_16way(sha_states, rmd_blocks);
    for (int i = 0; i < 16; i++) {
        rmd_state_ptrs[i] = rmd_states[i];
    }

    /* rmd_blocks[16][64] is contiguous, use gather-optimized variant */
    ripemd160_compress_avx512_contig(rmd_state_ptrs, (const uint8_t *)rmd_blocks, 64);

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
