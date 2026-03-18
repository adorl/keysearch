/*
 * hash_utils_avx512.c
 * AVX-512 specialized 16-way parallel hash160 function implementation
 */
#include "hash_utils.h"
#include <string.h>
#include <immintrin.h>

#ifdef __AVX512F__

/* Forward declaration of sha256_compress_avx512 SoA variant */
void sha256_compress_avx512_soa(__m512i soa_state[8], const uint8_t *blocks[16]);
/* Forward declaration of ripemd160_compress_avx512 SoA variant */
void ripemd160_compress_avx512_soa(__m512i soa_state[5], const __m512i w[16]);

/*
 * sha256_soa_to_rmd160_words — Bridge: convert SHA256 SoA state to RIPEMD160 message words.
 *
 * Performs big-endian→little-endian byte-swap on sha_state[0..7] using
 * _mm512_shuffle_epi8, then fills w[8..15] with the fixed RIPEMD160 padding
 * for a 32-byte message (0x80 padding marker, zeros, LE64(256) bit length).
 *
 * This replaces sha256_state_to_bytes_16way + load_le32_contig entirely,
 * keeping all data in SIMD registers.
 */
static inline __attribute__((always_inline)) void
sha256_soa_to_rmd160_words(const __m512i sha_state[8], __m512i w[16])
{
    /* Byte-swap mask: reverse bytes within each 32-bit lane (big-endian → little-endian) */
    const __m512i bswap = _mm512_set_epi8(
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3);

    /* SHA256 outputs 8 big-endian uint32 words → byte-swap for RIPEMD160 LE input */
    w[0] = _mm512_shuffle_epi8(sha_state[0], bswap);
    w[1] = _mm512_shuffle_epi8(sha_state[1], bswap);
    w[2] = _mm512_shuffle_epi8(sha_state[2], bswap);
    w[3] = _mm512_shuffle_epi8(sha_state[3], bswap);
    w[4] = _mm512_shuffle_epi8(sha_state[4], bswap);
    w[5] = _mm512_shuffle_epi8(sha_state[5], bswap);
    w[6] = _mm512_shuffle_epi8(sha_state[6], bswap);
    w[7] = _mm512_shuffle_epi8(sha_state[7], bswap);

    /* RIPEMD160 padding for 32-byte message:
     * w[8]  = 0x00000080 (0x80 marker byte at position 32, little-endian uint32)
     * w[9..13] = 0x00000000
     * w[14] = 0x00000100 (bit length 256 = 0x100, LE)
     * w[15] = 0x00000000
     */
    w[8]  = _mm512_set1_epi32(0x00000080);
    w[9]  = _mm512_setzero_si512();
    w[10] = _mm512_setzero_si512();
    w[11] = _mm512_setzero_si512();
    w[12] = _mm512_setzero_si512();
    w[13] = _mm512_setzero_si512();
    w[14] = _mm512_set1_epi32(0x00000100);
    w[15] = _mm512_setzero_si512();
}

/*
 * rmd160_soa_to_bytes_16way — Convert RIPEMD160 SoA state to 16 hash160 byte arrays.
 *
 * On x86 (native little-endian), RIPEMD160 state words are already in the
 * correct byte order.  We store each __m512i to a temp buffer and scatter
 * the 4-byte words into the appropriate positions of each lane's 20-byte output.
 */
static void rmd160_soa_to_bytes_16way(const __m512i soa_state[5], uint8_t hash160s[16][20])
{
    uint32_t tmp[16] __attribute__((aligned(64)));

    for (int w = 0; w < 5; w++) {
        _mm512_store_si512((__m512i *)tmp, soa_state[w]);
        int off = w * 4;
        for (int i = 0; i < 16; i++) {
            memcpy(&hash160s[i][off], &tmp[i], 4);
        }
    }
}

/*
 * hash160_16way_finalize_from_sha_soa — Full SoA pipeline finalization.
 *
 * Takes SHA256 SoA state directly, converts to RIPEMD160 message words in
 * registers, runs RIPEMD160 preloaded compression, and extracts hash160 bytes.
 * Eliminates sha256_state_to_bytes_16way, load_le32_contig, and rmd_store_16way.
 */
__attribute__((target("avx512f,avx512bw")))
void hash160_16way_finalize_from_sha_soa(__m512i sha_soa_state[8], uint8_t hash160s[16][20])
{
    /* Bridge: SHA256 SoA state -> RIPEMD160 pre-loaded message words */
    __m512i rmd_w[16];
    sha256_soa_to_rmd160_words(sha_soa_state, rmd_w);

    /* Initialize RIPEMD160 SoA state */
    __m512i rmd_soa[5];
    rmd_soa[0] = _mm512_set1_epi32(0x67452301);
    rmd_soa[1] = _mm512_set1_epi32((int)0xEFCDAB89);
    rmd_soa[2] = _mm512_set1_epi32((int)0x98BADCFE);
    rmd_soa[3] = _mm512_set1_epi32(0x10325476);
    rmd_soa[4] = _mm512_set1_epi32((int)0xC3D2E1F0);

    /* RIPEMD160 compression with pre-loaded words (no gather, no scatter) */
    ripemd160_compress_avx512_soa(rmd_soa, rmd_w);

    /* Convert RIPEMD160 SoA state to 16 x 20-byte hash160 */
    rmd160_soa_to_bytes_16way(rmd_soa, hash160s);
}

static void hash160_16way_prepadded_sha(const uint8_t *blocks1[16],
                                        const uint8_t *blocks2[16],
                                        uint8_t hash160s[16][20])
{
    /* Initialize SHA256 SoA state with standard IV */
    __m512i soa_state[8];
    soa_state[0] = _mm512_set1_epi32(0x6a09e667);
    soa_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
    soa_state[2] = _mm512_set1_epi32(0x3c6ef372);
    soa_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
    soa_state[4] = _mm512_set1_epi32(0x510e527f);
    soa_state[5] = _mm512_set1_epi32((int)0x9b05688c);
    soa_state[6] = _mm512_set1_epi32(0x1f83d9ab);
    soa_state[7] = _mm512_set1_epi32(0x5be0cd19);

    sha256_compress_avx512_soa(soa_state, blocks1);

    if (blocks2 != NULL) {
        sha256_compress_avx512_soa(soa_state, blocks2);
    }

    hash160_16way_finalize_from_sha_soa(soa_state, hash160s);
}

/*
 * 16-way parallel hash160 for compressed public keys (pre-padded, zero-copy)
 * blocks[16]: array of 64-byte block pointers with SHA256 padding already applied in-place by caller
 */
__attribute__((target("avx512f,avx512bw")))
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
__attribute__((target("avx512f,avx512bw")))
void hash160_16way_uncompressed_prepadded(const uint8_t *bufs[16], uint8_t hash160s[16][20])
{
    const uint8_t *blocks2[16];

    for (int i = 0; i < 16; i++) {
        blocks2[i] = bufs[i] + 64;
    }

    hash160_16way_prepadded_sha(bufs, blocks2, hash160s);
}

#ifdef __AVX512IFMA__

#include "secp256k1_keygen.h"

/*
 * hash160_16way_from_fe_soa — Compute hash160 directly from SoA field elements (Phase 1+2 optimization).
 *
 * Data flow (fully SIMD, no byte-array intermediate):
 *   fe_16x -> fe_to_sha256_words_16way -> assemble SHA-256 message words
 *   -> sha256_compress_avx512_soa_preloaded -> hash160_16way_finalize_from_sha_soa
 *
 * Compared to the old path:
 *   fe_16x -> fe_16x_store -> fe_get_b32_16way -> memcpy+padding -> load_be32_16way
 *   -> sha256_compress_avx512_soa -> hash160_16way_finalize_from_sha_soa
 *
 * Eliminates fe_get_b32_16way (6.72%) + load_be32_16way (3.63%) + memcpy overhead.
 *
 * == Key design: mapping between pubkey bytes and SHA-256 words ==
 *
 * Compressed pubkey (33 bytes = 1 block, 16 words):
 *   byte layout: [prefix(1)] [X(32)]
 *   word layout: w[0] = prefix|X[0:2], w[1..7] = X[3:31] (cross-byte-boundary splicing), w[8] = 0x80 marker, w[9..13] = 0, w[14..15] = bit_length
 *
 *   Specifically, the 8 SHA-256 words of the X coordinate (xw[0..7], pure X without prefix) map as:
 *   - w[0] = (prefix << 24) | (xw[0] >> 8)        <- prefix byte occupies the top 8 bits
 *   - w[1] = (xw[0] << 24) | (xw[1] >> 8)
 *   - w[2] = (xw[1] << 24) | (xw[2] >> 8)
 *   - ...
 *   - w[7] = (xw[6] << 24) | (xw[7] >> 8)
 *   - w[8] = (xw[7] << 24) | 0x00800000           <- last byte of X + 0x80 padding marker
 *   - w[9..13] = 0
 *   - w[14] = 0x00000000, w[15] = 0x00000108       <- bit length = 33*8 = 264
 *
 * Uncompressed pubkey (65 bytes = 2 blocks, 32 words):
 *   byte layout: [0x04(1)] [X(32)] [Y(32)]
 *   Block 1 (64 bytes): [0x04] [X(32)] [Y[0:30](31)]
 *   Block 2 (64 bytes): [Y[31](1)] [0x80] [zeros(54)] [bit_length(8)]
 *
 *   - b1_w[0] = 0x04000000 | (xw[0] >> 8)
 *   - b1_w[1..7] = (xw[i-1] << 24) | (xw[i] >> 8)    (same as compressed)
 *   - b1_w[8] = (xw[7] << 24) | (yw[0] >> 8)
 *   - b1_w[9..14] = (yw[i-8-1] << 24) | (yw[i-8] >> 8)
 *   - b1_w[15] = (yw[6] << 24) | (yw[7] >> 8)
 *   - b2_w[0] = (yw[7] << 24) | 0x00800000            <- last byte of Y + 0x80
 *   - b2_w[1..13] = 0
 *   - b2_w[14] = 0x00000000, b2_w[15] = 0x00000208    <- bit length = 65*8 = 520
 */
__attribute__((target("avx512f,avx512bw,avx512ifma")))
void hash160_16way_from_fe_soa(const secp256k1_fe_16x *fe_x,
                               const secp256k1_fe_16x *fe_y,
                               uint8_t hash160_comp[16][20],
                               uint8_t hash160_uncomp[16][20])
{
    /* Generate SHA-256 big-endian words directly from SoA limbs */
    __m512i xw[8], yw[8];
    fe_to_sha256_words_16way(fe_x, xw);
    fe_to_sha256_words_16way(fe_y, yw);

    /*
     * Detect Y coordinate parity: y_is_odd = bit 0 of n[0].
     * fe_16x.n[0][0] has 8 64-bit lanes, n[0][1] has 8 64-bit lanes.
     * Extract bit 0 and pack into a 16-bit mask.
     */
    __m512i one64 = _mm512_set1_epi64(1);
    __m512i y_lo_bits = _mm512_and_si512(fe_y->n[0][0], one64);  /* 8 × (0 or 1) */
    __m512i y_hi_bits = _mm512_and_si512(fe_y->n[0][1], one64);
    /* Convert to mask: lane != 0 -> bit set */
    __mmask8 odd_lo = _mm512_test_epi64_mask(y_lo_bits, y_lo_bits);
    __mmask8 odd_hi = _mm512_test_epi64_mask(y_hi_bits, y_hi_bits);

    /*
     * Construct compressed prefix word:
     * Each lane's prefix = 0x02 (even) or 0x03 (odd)
     * prefix_word = prefix << 24 = 0x02000000 or 0x03000000
     *
     * Since the prefix is in 32-bit lanes but the mask is 8-bit (corresponding to 64-bit lanes),
     * we need to expand: lo mask bit i -> 32-bit lane i (in 512-bit)
     *                    hi mask bit i -> 32-bit lane i+8
     */
    __m512i prefix_even = _mm512_set1_epi32(0x02000000);
    __m512i prefix_odd  = _mm512_set1_epi32(0x03000000);
    /* Expand 8-bit 64-lane mask to 16-bit 32-lane mask:
     * lo half: bits 0-7 correspond to 32-bit lanes 0,2,4,...,14 (each 64-bit lane contains 2 32-bit lanes)
     * However, since _mm512_cvtepi64_epi32 has already packed each 64-bit lane into a single 32-bit lane,
     * the 16 32-bit lanes in xw/yw are tightly arranged (lo 8 + hi 8).
     * Therefore, simply use lo mask | (hi mask << 8). */
    __mmask16 odd_mask16 = (__mmask16)odd_lo | ((__mmask16)odd_hi << 8);
    __m512i comp_prefix = _mm512_mask_mov_epi32(prefix_even, odd_mask16, prefix_odd);

    /*
     * === Compressed pubkey SHA-256 (1 block, 16 words) ===
     *
     * pubkey bytes: [prefix(1)] [X(32)] [padding]
     * Since the prefix occupies 1 byte, all X bytes are shifted right by 1 byte (8 bits).
     * w[i] = (prev << 24) | (cur >> 8)
     */
    __m512i comp_w[16];
    comp_w[0]  = _mm512_or_si512(comp_prefix, _mm512_srli_epi32(xw[0], 8));
    comp_w[1]  = _mm512_or_si512(_mm512_slli_epi32(xw[0], 24), _mm512_srli_epi32(xw[1], 8));
    comp_w[2]  = _mm512_or_si512(_mm512_slli_epi32(xw[1], 24), _mm512_srli_epi32(xw[2], 8));
    comp_w[3]  = _mm512_or_si512(_mm512_slli_epi32(xw[2], 24), _mm512_srli_epi32(xw[3], 8));
    comp_w[4]  = _mm512_or_si512(_mm512_slli_epi32(xw[3], 24), _mm512_srli_epi32(xw[4], 8));
    comp_w[5]  = _mm512_or_si512(_mm512_slli_epi32(xw[4], 24), _mm512_srli_epi32(xw[5], 8));
    comp_w[6]  = _mm512_or_si512(_mm512_slli_epi32(xw[5], 24), _mm512_srli_epi32(xw[6], 8));
    comp_w[7]  = _mm512_or_si512(_mm512_slli_epi32(xw[6], 24), _mm512_srli_epi32(xw[7], 8));
    /* w[8] = (xw[7] low 8 bits shifted to upper 24 bits) | 0x00800000 (0x80 padding marker) */
    comp_w[8]  = _mm512_or_si512(_mm512_slli_epi32(xw[7], 24), _mm512_set1_epi32(0x00800000));
    comp_w[9]  = _mm512_setzero_si512();
    comp_w[10] = _mm512_setzero_si512();
    comp_w[11] = _mm512_setzero_si512();
    comp_w[12] = _mm512_setzero_si512();
    comp_w[13] = _mm512_setzero_si512();
    comp_w[14] = _mm512_setzero_si512();
    /* bit length = 33*8 = 264 = 0x108 */
    comp_w[15] = _mm512_set1_epi32(0x00000108);

    /* SHA-256 compress (compressed pubkey, 1 block) */
    __m512i comp_sha_state[8];
    comp_sha_state[0] = _mm512_set1_epi32(0x6a09e667);
    comp_sha_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
    comp_sha_state[2] = _mm512_set1_epi32(0x3c6ef372);
    comp_sha_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
    comp_sha_state[4] = _mm512_set1_epi32(0x510e527f);
    comp_sha_state[5] = _mm512_set1_epi32((int)0x9b05688c);
    comp_sha_state[6] = _mm512_set1_epi32(0x1f83d9ab);
    comp_sha_state[7] = _mm512_set1_epi32(0x5be0cd19);
    sha256_compress_avx512_soa_preloaded(comp_sha_state, comp_w);

    /* Finalize: SHA-256 state → RIPEMD-160 → hash160 */
    hash160_16way_finalize_from_sha_soa(comp_sha_state, hash160_comp);

    /*
     * === Uncompressed pubkey SHA-256 (2 blocks, 32 words) ===
     *
     * pubkey bytes: [0x04(1)] [X(32)] [Y(32)]
     * Block 1 (64 bytes): [0x04] [X(32)] [Y[0:30](31)]
     * Block 2 (64 bytes): [Y[31](1)] [0x80] [zeros(54)] [bit_length(8)]
     *
     * Similarly, all coordinate bytes are shifted right by 1 byte.
     */
    __m512i uncomp_prefix = _mm512_set1_epi32(0x04000000);

    /* Block 1: [0x04|X[0:2]] [X bytes shift] ... [X|Y shift] ... */
    __m512i b1_w[16];
    b1_w[0]  = _mm512_or_si512(uncomp_prefix, _mm512_srli_epi32(xw[0], 8));
    b1_w[1]  = _mm512_or_si512(_mm512_slli_epi32(xw[0], 24), _mm512_srli_epi32(xw[1], 8));
    b1_w[2]  = _mm512_or_si512(_mm512_slli_epi32(xw[1], 24), _mm512_srli_epi32(xw[2], 8));
    b1_w[3]  = _mm512_or_si512(_mm512_slli_epi32(xw[2], 24), _mm512_srli_epi32(xw[3], 8));
    b1_w[4]  = _mm512_or_si512(_mm512_slli_epi32(xw[3], 24), _mm512_srli_epi32(xw[4], 8));
    b1_w[5]  = _mm512_or_si512(_mm512_slli_epi32(xw[4], 24), _mm512_srli_epi32(xw[5], 8));
    b1_w[6]  = _mm512_or_si512(_mm512_slli_epi32(xw[5], 24), _mm512_srli_epi32(xw[6], 8));
    b1_w[7]  = _mm512_or_si512(_mm512_slli_epi32(xw[6], 24), _mm512_srli_epi32(xw[7], 8));
    /* X->Y transition: w[8] = (xw[7] << 24) | (yw[0] >> 8) */
    b1_w[8]  = _mm512_or_si512(_mm512_slli_epi32(xw[7], 24), _mm512_srli_epi32(yw[0], 8));
    b1_w[9]  = _mm512_or_si512(_mm512_slli_epi32(yw[0], 24), _mm512_srli_epi32(yw[1], 8));
    b1_w[10] = _mm512_or_si512(_mm512_slli_epi32(yw[1], 24), _mm512_srli_epi32(yw[2], 8));
    b1_w[11] = _mm512_or_si512(_mm512_slli_epi32(yw[2], 24), _mm512_srli_epi32(yw[3], 8));
    b1_w[12] = _mm512_or_si512(_mm512_slli_epi32(yw[3], 24), _mm512_srli_epi32(yw[4], 8));
    b1_w[13] = _mm512_or_si512(_mm512_slli_epi32(yw[4], 24), _mm512_srli_epi32(yw[5], 8));
    b1_w[14] = _mm512_or_si512(_mm512_slli_epi32(yw[5], 24), _mm512_srli_epi32(yw[6], 8));
    b1_w[15] = _mm512_or_si512(_mm512_slli_epi32(yw[6], 24), _mm512_srli_epi32(yw[7], 8));

    /* Block 2: [Y last byte | 0x80 | zeros | bitlen] */
    __m512i b2_w[16];
    /* b2_w[0] = (yw[7] << 24) | 0x00800000 (last byte of Y + 0x80) */
    b2_w[0]  = _mm512_or_si512(_mm512_slli_epi32(yw[7], 24), _mm512_set1_epi32(0x00800000));
    b2_w[1]  = _mm512_setzero_si512();
    b2_w[2]  = _mm512_setzero_si512();
    b2_w[3]  = _mm512_setzero_si512();
    b2_w[4]  = _mm512_setzero_si512();
    b2_w[5]  = _mm512_setzero_si512();
    b2_w[6]  = _mm512_setzero_si512();
    b2_w[7]  = _mm512_setzero_si512();
    b2_w[8]  = _mm512_setzero_si512();
    b2_w[9]  = _mm512_setzero_si512();
    b2_w[10] = _mm512_setzero_si512();
    b2_w[11] = _mm512_setzero_si512();
    b2_w[12] = _mm512_setzero_si512();
    b2_w[13] = _mm512_setzero_si512();
    b2_w[14] = _mm512_setzero_si512();
    /* bit length = 65*8 = 520 = 0x208 */
    b2_w[15] = _mm512_set1_epi32(0x00000208);

    /* SHA-256 compress (uncompressed pubkey, 2 blocks) */
    __m512i uncomp_sha_state[8];
    uncomp_sha_state[0] = _mm512_set1_epi32(0x6a09e667);
    uncomp_sha_state[1] = _mm512_set1_epi32((int)0xbb67ae85);
    uncomp_sha_state[2] = _mm512_set1_epi32(0x3c6ef372);
    uncomp_sha_state[3] = _mm512_set1_epi32((int)0xa54ff53a);
    uncomp_sha_state[4] = _mm512_set1_epi32(0x510e527f);
    uncomp_sha_state[5] = _mm512_set1_epi32((int)0x9b05688c);
    uncomp_sha_state[6] = _mm512_set1_epi32(0x1f83d9ab);
    uncomp_sha_state[7] = _mm512_set1_epi32(0x5be0cd19);
    sha256_compress_avx512_soa_preloaded(uncomp_sha_state, b1_w);
    sha256_compress_avx512_soa_preloaded(uncomp_sha_state, b2_w);

    /* Finalize: SHA-256 state → RIPEMD-160 → hash160 */
    hash160_16way_finalize_from_sha_soa(uncomp_sha_state, hash160_uncomp);
}

#endif /* __AVX512IFMA__ */

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

