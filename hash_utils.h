#ifndef HASH_UTILS_H
#define HASH_UTILS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Convert private key byte array to hex string */
void bytes_to_hex(const uint8_t *bytes, int len, char *hex_out);

/* Double SHA256 hash */
void sha256d(const uint8_t *data, size_t len, uint8_t *out);

/* Base58Check encoding */
void base58check_encode(const uint8_t *payload, size_t payload_len, char *out);

/*
 * Base58Check decoding: decode a Bitcoin address string back to a 20-byte RIPEMD160 hash
 * b58str      : input Base58Check encoded string
 * hash160_out : output buffer, at least 20 bytes; pass NULL to only validate
 * Return value:
 *   0  : success, hash160 written to hash160_out
 *  -1  : format error (invalid character, insufficient length, buffer overflow)
 *  -2  : checksum mismatch
 */
int base58check_decode(const char *b58str, uint8_t *hash160_out);

/*
 * Compute hash160 (SHA256 -> RIPEMD160) for both compressed and uncompressed public keys
 * from a 32-byte private key.
 * compressed_hash160   : output for compressed pubkey hash160 (20 bytes), pass NULL to skip
 * uncompressed_hash160 : output for uncompressed pubkey hash160 (20 bytes), pass NULL to skip
 * Return value: 0 on success, -1 on failure (pubkey generation failed)
 */
int privkey_to_hash160(const uint8_t *privkey,
                       uint8_t *compressed_hash160,
                       uint8_t *uncompressed_hash160);

/*
 * Compute hash160 (SHA256 -> RIPEMD160) directly from serialized public key bytes
 * pubkey_bytes : serialized public key bytes (33 bytes compressed or 65 bytes uncompressed)
 * len          : length of public key bytes (33 or 65)
 * hash160_out  : output buffer, at least 20 bytes
 */
void pubkey_bytes_to_hash160(const uint8_t *pubkey_bytes, size_t len,
                              uint8_t *hash160_out);

/*
 * Compute both compressed and uncompressed Bitcoin addresses (P2PKH)
 * from a 32-byte private key simultaneously.
 * compressed_out   : output buffer for compressed address (at least ADDRESS_LEN+1 bytes)
 * uncompressed_out : output buffer for uncompressed address (at least ADDRESS_LEN+1 bytes)
 */
int privkey_to_address(const uint8_t *privkey,
                       char *compressed_out,
                       char *uncompressed_out);

#if defined(__AVX2__) || defined(__AVX512F__)
/*
 * 8-way parallel hash160 (SHA256->RIPEMD160) for compressed public keys (33 bytes)
 * pubkeys[8]    : 8 compressed public key byte pointers (33 bytes each)
 * hash160s[8]   : 8 output buffers (20 bytes each)
 */
void hash160_8way_compressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20]);

/*
 * 8-way parallel hash160 for compressed public keys (pre-padded, zero-copy)
 * blocks[8]     : 8 pointers to 64-byte blocks with SHA256 padding already applied
 * hash160s[8]   : 8 output buffers (20 bytes each)
 */
void hash160_8way_compressed_prepadded(const uint8_t *blocks[8], uint8_t hash160s[8][20]);

/*
 * 8-way parallel hash table lookup: search for 8 hash160 values simultaneously
 * h160s[8]: 8 hash160 pointers (20 bytes each)
 * Returns 8-bit hit mask: bit i set means lane i matched
 */
uint8_t ht_contains_8way(const uint8_t *h160s[8]);

/*
 * In-place SHA256 padded block construction for a 33-byte compressed public key
 * buf must be at least 64 bytes, pubkey data already written to buf[0..32],
 * this function fills in the padding
 */
static inline void sha256_pad_block_33(uint8_t buf[64])
{
    /* Message length 33 bytes, 0x80 padding, bit length = 33*8 = 264 = 0x108 */
    buf[33] = 0x80;
    buf[34] = 0x00; buf[35] = 0x00; buf[36] = 0x00; buf[37] = 0x00;
    buf[38] = 0x00; buf[39] = 0x00; buf[40] = 0x00; buf[41] = 0x00;
    buf[42] = 0x00; buf[43] = 0x00; buf[44] = 0x00; buf[45] = 0x00;
    buf[46] = 0x00; buf[47] = 0x00; buf[48] = 0x00; buf[49] = 0x00;
    buf[50] = 0x00; buf[51] = 0x00; buf[52] = 0x00; buf[53] = 0x00;
    buf[54] = 0x00; buf[55] = 0x00;
    /* Big-endian bit length 264 = 0x0000000000000108 */
    buf[56] = 0x00; buf[57] = 0x00; buf[58] = 0x00; buf[59] = 0x00;
    buf[60] = 0x00; buf[61] = 0x00; buf[62] = 0x01; buf[63] = 0x08;
}

/*
 * 8-way parallel hash160 (SHA256->RIPEMD160) for uncompressed public keys (65 bytes)
 * pubkeys[8]    : 8 uncompressed public key byte pointers (65 bytes each)
 * hash160s[8]   : 8 output buffers (20 bytes each)
 */
void hash160_8way_uncompressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20]);

/*
 * 8-way parallel hash160 for uncompressed public keys (pre-padded, zero-copy)
 * bufs[8]       : 8 pointers to 128-byte buffers with the following layout:
 *                   buf[0..63] = first 64 bytes of pubkey
 *                   buf[64..127]= SHA256 block2
 * hash160s[8]   : 8 output buffers (20 bytes each)
 */
void hash160_8way_uncompressed_prepadded(const uint8_t *bufs[8], uint8_t hash160s[8][20]);

/*
 * In-place SHA256 block2 construction for a 65-byte uncompressed public key
 * (writes to buf[64..127])
 * buf must be at least 128 bytes, pubkey data already written to buf[0..64],
 * this function fills in block2 at buf[64..127]
 * Before call: buf[0..64] = uncompressed public key (65 bytes)
 * After call:  buf[64..127] = SHA256 block2
 */
static inline void sha256_pad_block2_65(uint8_t buf[128])
{
    /* Read the 65th byte of the pubkey (buf[64]), construct block2 */
    uint8_t last_byte = buf[64];
    /* block2: last_byte + 0x80 + zero padding + 8-byte big-endian bit length (65*8=520=0x208) */
    buf[64]  = last_byte;
    buf[65]  = 0x80;
    buf[66]  = 0x00; buf[67]  = 0x00; buf[68]  = 0x00; buf[69]  = 0x00;
    buf[70]  = 0x00; buf[71]  = 0x00; buf[72]  = 0x00; buf[73]  = 0x00;
    buf[74]  = 0x00; buf[75]  = 0x00; buf[76]  = 0x00; buf[77]  = 0x00;
    buf[78]  = 0x00; buf[79]  = 0x00; buf[80]  = 0x00; buf[81]  = 0x00;
    buf[82]  = 0x00; buf[83]  = 0x00; buf[84]  = 0x00; buf[85]  = 0x00;
    buf[86]  = 0x00; buf[87]  = 0x00; buf[88]  = 0x00; buf[89]  = 0x00;
    buf[90]  = 0x00; buf[91]  = 0x00; buf[92]  = 0x00; buf[93]  = 0x00;
    buf[94]  = 0x00; buf[95]  = 0x00; buf[96]  = 0x00; buf[97]  = 0x00;
    buf[98]  = 0x00; buf[99]  = 0x00; buf[100] = 0x00; buf[101] = 0x00;
    buf[102] = 0x00; buf[103] = 0x00; buf[104] = 0x00; buf[105] = 0x00;
    buf[106] = 0x00; buf[107] = 0x00; buf[108] = 0x00; buf[109] = 0x00;
    buf[110] = 0x00; buf[111] = 0x00; buf[112] = 0x00; buf[113] = 0x00;
    buf[114] = 0x00; buf[115] = 0x00; buf[116] = 0x00; buf[117] = 0x00;
    buf[118] = 0x00; buf[119] = 0x00;
    /* Big-endian bit length 520 = 0x0000000000000208 */
    buf[120] = 0x00; buf[121] = 0x00; buf[122] = 0x00; buf[123] = 0x00;
    buf[124] = 0x00; buf[125] = 0x00; buf[126] = 0x02; buf[127] = 0x08;
}
#endif /* __AVX2__ || __AVX512F__ */

/*
 * Hash table slot structure:
 *   fp     : first 4 bytes fingerprint for fast filtering (0 means empty slot)
 *   h160   : full 20-byte hash160
 */
struct ht_slot {
    uint32_t fp;        /* 4-byte fingerprint (big-endian read) */
    uint8_t h160[20];  /* full 20-byte hash160 */
};

/* Global hash table (allocated by ht_init, freed by ht_free) */
extern struct ht_slot *ht_slots;
extern uint32_t ht_mask;   /* slot mask = slot count - 1 (slot count is a power of 2) */

/*
 * Initialize hash table: allocate capacity slots (capacity must be a power of 2)
 * Returns 0 on success, -1 on failure
 */
int ht_init(uint32_t capacity);

/* Free hash table memory */
void ht_free(void);

/*
 * Insert hash160 (20 bytes) into hash table
 * Uses linear probing for collision resolution
 */
void ht_insert(const uint8_t *h160);

/*
 * Look up hash160 (20 bytes) in hash table
 * Returns 1 on hit, 0 on miss
 */
int ht_contains(const uint8_t *h160);

#ifdef __AVX512F__
/*
 * 16-way parallel hash160 (SHA256->RIPEMD160) for compressed public keys (33 bytes)
 * pubkeys[16]   : 16 compressed public key byte pointers (33 bytes each)
 * hash160s[16]  : 16 output buffers (20 bytes each)
 */
void hash160_16way_compressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20]);

/*
 * 16-way parallel hash160 (SHA256->RIPEMD160) for uncompressed public keys (65 bytes)
 * pubkeys[16]   : 16 uncompressed public key byte pointers (65 bytes each)
 * hash160s[16]  : 16 output buffers (20 bytes each)
 */
void hash160_16way_uncompressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20]);

/*
 * 16-way parallel hash160 for compressed public keys (pre-padded, zero-copy)
 * blocks[16]    : 16 pointers to 64-byte blocks with SHA256 padding already applied
 * hash160s[16]  : 16 output buffers (20 bytes each)
 */
void hash160_16way_compressed_prepadded(const uint8_t *blocks[16], uint8_t hash160s[16][20]);

/*
 * 16-way parallel hash160 for uncompressed public keys (pre-padded, zero-copy)
 * bufs[16]      : 16 pointers to 128-byte buffers with the following layout:
 *                   buf[0..63]  = first 64 bytes of pubkey (SHA256 block1, no padding needed)
 *                   buf[64..127]= SHA256 block2 (padding already applied)
 * hash160s[16]  : 16 output buffers (20 bytes each)
 */
void hash160_16way_uncompressed_prepadded(const uint8_t *bufs[16], uint8_t hash160s[16][20]);

/*
 * 16-way parallel hash table lookup: search for 16 hash160 values simultaneously
 * h160s[16]: 16 hash160 pointers (20 bytes each)
 * Returns 16-bit hit mask: bit i set means lane i matched
 */
uint16_t ht_contains_16way(const uint8_t *h160s[16]);
#endif /* __AVX512F__ */

#ifdef __cplusplus
}
#endif

#endif /* HASH_UTILS_H */


