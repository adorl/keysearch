
#ifndef BECH32_H
#define BECH32_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bech32 decode: decode a bech32/bech32m encoded string into a witness program
 *
 * addr         : input bech32 address string (e.g. "bc1q...")
 * witness_ver  : output witness version (0~16)
 * witness_prog : output witness program byte array, at least 40 bytes
 * witness_len  : output actual length of the witness program
 *
 * Return value:
 *   0  : success
 *  -1  : format error (invalid character, illegal length, no separator, etc.)
 *  -2  : checksum error
 *  -3  : invalid witness version or length
 */
int bech32_decode_witness(const char *addr,
                          int *witness_ver,
                          uint8_t *witness_prog,
                          size_t *witness_len);

/*
 * Bech32 encode: encode witness version and witness program into a bech32 address string
 *
 * hrp          : human-readable part (e.g. "bc")
 * witness_ver  : witness version (0~16)
 * witness_prog : witness program byte array
 * witness_len  : witness program length
 * output       : output buffer, at least 90 bytes
 *
 * Return value:
 *   0  : success
 *  -1  : invalid parameters
 */
int bech32_encode_witness(const char *hrp,
                          int witness_ver,
                          const uint8_t *witness_prog,
                          size_t witness_len,
                          char *output);

#ifdef __cplusplus
}
#endif

#endif /* BECH32_H */

