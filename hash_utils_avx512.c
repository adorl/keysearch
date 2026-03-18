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

