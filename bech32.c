
/*
 * bech32.c
 * Pure C implementation of Bech32/Bech32m encoding and decoding
 * Reference: BIP173 (bech32) and BIP350 (bech32m)
 */

#include "bech32.h"
#include <string.h>
#include <ctype.h>

/* Bech32 character set */
static const char BECH32_CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/* Bech32 character reverse lookup table (-1 means invalid character) */
static const int8_t BECH32_CHARSET_REV[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    15, -1, 10, 17, 21, 20, 26, 30,  7,  5, -1, -1, -1, -1, -1, -1,
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,
    -1, 29, -1, 24, 13, 25,  9,  8, 23, -1, 18, 22, 31, 27, 19, -1,
     1,  0,  3, 16, 11, 28, 12, 14,  6,  4,  2, -1, -1, -1, -1, -1,
};

/* Bech32 checksum constants */
#define BECH32_CONST  1
#define BECH32M_CONST 0x2bc830a3

/* Bech32 polynomial modular arithmetic */
static uint32_t bech32_polymod(const uint8_t *values, size_t len)
{
    uint32_t chk = 1;
    for (size_t i = 0; i < len; i++) {
        uint8_t top = chk >> 25;
        chk = ((chk & 0x1ffffff) << 5) ^ values[i];
        if (top & 1)  chk ^= 0x3b6a57b2;
        if (top & 2)  chk ^= 0x26508e6d;
        if (top & 4)  chk ^= 0x1ea119fa;
        if (top & 8)  chk ^= 0x3d4233dd;
        if (top & 16) chk ^= 0x2a1462b3;
    }
    return chk;
}

/* Expand HRP for checksum computation */
static void bech32_hrp_expand(const char *hrp, size_t hrp_len, uint8_t *out)
{
    for (size_t i = 0; i < hrp_len; i++)
        out[i] = (uint8_t)(hrp[i] >> 5);
    out[hrp_len] = 0;
    for (size_t i = 0; i < hrp_len; i++)
        out[hrp_len + 1 + i] = (uint8_t)(hrp[i] & 0x1f);
}

/* Verify bech32 checksum, returns encoding type constant (BECH32_CONST or BECH32M_CONST), returns 0 on failure */
static uint32_t bech32_verify_checksum(const char *hrp, size_t hrp_len,
                                       const uint8_t *data, size_t data_len)
{
    /* Construct polymod input: hrp_expand + data */
    size_t expand_len = hrp_len * 2 + 1;
    size_t total_len = expand_len + data_len;
    uint8_t values[256]; /* sufficiently large buffer */
    if (total_len > sizeof(values))
        return 0;

    bech32_hrp_expand(hrp, hrp_len, values);
    memcpy(values + expand_len, data, data_len);

    uint32_t polymod = bech32_polymod(values, total_len);
    if (polymod == BECH32_CONST)
        return BECH32_CONST;
    if (polymod == BECH32M_CONST)
        return BECH32M_CONST;
    return 0;
}

/* Create bech32 checksum */
static void bech32_create_checksum(const char *hrp, size_t hrp_len,
                                   const uint8_t *data, size_t data_len,
                                   uint32_t encoding_const,
                                   uint8_t *checksum_out)
{
    size_t expand_len = hrp_len * 2 + 1;
    size_t total_len = expand_len + data_len + 6;
    uint8_t values[256];

    bech32_hrp_expand(hrp, hrp_len, values);
    memcpy(values + expand_len, data, data_len);
    memset(values + expand_len + data_len, 0, 6);

    uint32_t polymod = bech32_polymod(values, total_len) ^ encoding_const;
    for (int i = 0; i < 6; i++)
        checksum_out[i] = (uint8_t)((polymod >> (5 * (5 - i))) & 0x1f);
}

/* Convert between bit groups (generic bit conversion) */
static int convert_bits(uint8_t *out, size_t *out_len,
                        int to_bits,
                        const uint8_t *in, size_t in_len,
                        int from_bits,
                        int pad)
{
    uint32_t acc = 0;
    int bits = 0;
    size_t max_out = *out_len;
    size_t pos = 0;

    for (size_t i = 0; i < in_len; i++) {
        uint8_t value = in[i];
        if ((value >> from_bits) != 0)
            return -1;
        acc = (acc << from_bits) | value;
        bits += from_bits;
        while (bits >= to_bits) {
            bits -= to_bits;
            if (pos >= max_out)
                return -1;
            out[pos++] = (uint8_t)((acc >> bits) & ((1 << to_bits) - 1));
        }
    }

    if (pad) {
        if (bits > 0) {
            if (pos >= max_out)
                return -1;
            out[pos++] = (uint8_t)((acc << (to_bits - bits)) & ((1 << to_bits) - 1));
        }
    } else {
        if (bits >= from_bits)
            return -1;
        if ((acc << (to_bits - bits)) & ((1 << to_bits) - 1))
            return -1;
    }

    *out_len = pos;
    return 0;
}

/* Convert 8-bit byte array to 5-bit array */
static int convert_8to5(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len)
{
    return convert_bits(out, out_len, 5, in, in_len, 8, 1);
}

/* Convert 5-bit array to 8-bit byte array */
static int convert_5to8(const uint8_t *in, size_t in_len,
                        uint8_t *out, size_t *out_len)
{
    return convert_bits(out, out_len, 8, in, in_len, 5, 0);
}

int bech32_decode_witness(const char *addr,
                          int *witness_ver,
                          uint8_t *witness_prog,
                          size_t *witness_len)
{
    size_t addr_len = strlen(addr);
    if (addr_len < 8 || addr_len > 90)
        return -1;

    /* Find the last '1' separator */
    int sep_pos = -1;
    for (int i = (int)addr_len - 1; i >= 0; i--) {
        if (addr[i] == '1') {
            sep_pos = i;
            break;
        }
    }
    if (sep_pos < 1 || sep_pos + 7 > (int)addr_len)
        return -1;

    /* Extract HRP (convert to lowercase) */
    char hrp[84];
    for (int i = 0; i < sep_pos; i++) {
        hrp[i] = (char)tolower((unsigned char)addr[i]);
    }
    hrp[sep_pos] = '\0';
    size_t hrp_len = (size_t)sep_pos;

    /* Check if HRP is "bc" (Bitcoin mainnet) */
    if (strcmp(hrp, "bc") != 0)
        return -1;

    /* Decode data part */
    size_t data_len = addr_len - sep_pos - 1;
    uint8_t data[90];
    int has_lower = 0, has_upper = 0;

    for (size_t i = 0; i < data_len; i++) {
        char c = addr[sep_pos + 1 + i];
        if (c >= 'a' && c <= 'z') has_lower = 1;
        if (c >= 'A' && c <= 'Z') has_upper = 1;
        c = (char)tolower((unsigned char)c);

        if (c < 33 || c > 126)
            return -1;
        int8_t val = BECH32_CHARSET_REV[(unsigned char)c];
        if (val == -1)
            return -1;
        data[i] = (uint8_t)val;
    }

    /* Mixed case not allowed */
    if (has_lower && has_upper)
        return -1;

    /* Verify checksum */
    uint32_t encoding = bech32_verify_checksum(hrp, hrp_len, data, data_len);
    if (encoding == 0)
        return -2;

    /* Extract witness version */
    int wver = data[0];
    if (wver > 16)
        return -3;

    /* witness version 0 must use bech32, version 1+ must use bech32m */
    if (wver == 0 && encoding != BECH32_CONST)
        return -2;
    if (wver > 0 && encoding != BECH32M_CONST)
        return -2;

    /* Convert 5-bit data to 8-bit witness program (strip version byte and 6-byte checksum) */
    size_t prog_5bit_len = data_len - 1 - 6; /* strip witness_ver and 6-byte checksum */
    size_t prog_len = 40;
    if (convert_5to8(data + 1, prog_5bit_len, witness_prog, &prog_len) != 0)
        return -3;

    /* Check witness program length */
    if (prog_len < 2 || prog_len > 40)
        return -3;
    if (wver == 0 && prog_len != 20 && prog_len != 32)
        return -3;

    *witness_ver = wver;
    *witness_len = prog_len;
    return 0;
}

int bech32_encode_witness(const char *hrp,
                          int witness_ver,
                          const uint8_t *witness_prog,
                          size_t witness_len,
                          char *output)
{
    if (witness_ver < 0 || witness_ver > 16)
        return -1;
    if (witness_len < 2 || witness_len > 40)
        return -1;

    /* Convert 8-bit witness program to 5-bit */
    uint8_t prog_5bit[65];
    size_t prog_5bit_len = sizeof(prog_5bit);
    if (convert_8to5(witness_prog, witness_len, prog_5bit, &prog_5bit_len) != 0)
        return -1;

    /* Construct data: witness_ver + prog_5bit */
    uint8_t data[70];
    data[0] = (uint8_t)witness_ver;
    memcpy(data + 1, prog_5bit, prog_5bit_len);
    size_t data_len = 1 + prog_5bit_len;

    /* Compute checksum */
    uint32_t encoding_const = (witness_ver == 0) ? BECH32_CONST : BECH32M_CONST;
    uint8_t checksum[6];
    size_t hrp_len = strlen(hrp);
    bech32_create_checksum(hrp, hrp_len, data, data_len, encoding_const, checksum);

    /* Construct output string: hrp + '1' + data_chars + checksum_chars */
    size_t pos = 0;
    for (size_t i = 0; i < hrp_len; i++)
        output[pos++] = hrp[i];
    output[pos++] = '1';
    for (size_t i = 0; i < data_len; i++)
        output[pos++] = BECH32_CHARSET[data[i]];
    for (int i = 0; i < 6; i++)
        output[pos++] = BECH32_CHARSET[checksum[i]];
    output[pos] = '\0';

    return 0;
}

