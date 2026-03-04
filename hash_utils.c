#include "hash_utils.h"
#include "sha256.h"
#include "ripemd160.h"
#include <string.h>
#include <stdio.h>
#ifndef USE_PUBKEY_API_ONLY
#  include "secp256k1_keygen.h"
#else
#  include <secp256k1.h>
#  include "secp256k1_keygen.h"
#endif

/* secp256k1上下文 */
extern secp256k1_context *secp_ctx;

/* Base58字符表 */
const char BASE58_CHARS[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* Base58反向映射表：ASCII->Base58索引（-1表示非法字符） */
static int base58_decode_map[256];
static int base58_decode_map_inited = 0;

/* 私钥字节数组转十六进制字符串 */
void bytes_to_hex(const uint8_t *bytes, int len, char *hex_out)
{
    for (int i = 0; i < len; i++) {
        sprintf(hex_out + i * 2, "%02x", bytes[i]);
    }
    hex_out[len * 2] = '\0';
}

/* SHA256两次哈希 */
void sha256d(const uint8_t *data, size_t len, uint8_t *out)
{
    uint8_t tmp[32];
    sha256(data, len, tmp);
    sha256(tmp, 32, out);
}

/* Base58Check编码 */
void base58check_encode(const uint8_t *payload, size_t payload_len, char *out)
{
    /* 计算校验和：sha256d前4字节 */
    uint8_t checksum[32];
    sha256d(payload, payload_len, checksum);

    /* 拼接 payload + checksum[0..3] */
    uint8_t buf[payload_len + 4];
    memcpy(buf, payload, payload_len);
    memcpy(buf + payload_len, checksum, 4);
    size_t total = payload_len + 4;

    /* 统计前导零字节 */
    int leading_zeros = 0;
    for (size_t i = 0; i < total && buf[i] == 0; i++)
        leading_zeros++;

    /* 大数转Base58 */
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

    /* 构建输出字符串（逆序） */
    int pos = 0;
    for (int i = 0; i < leading_zeros; i++)
        out[pos++] = '1';
    for (int i = digits_len - 1; i >= 0; i--)
        out[pos++] = BASE58_CHARS[digits[i]];
    out[pos] = '\0';
}

/* 初始化反向映射表（只执行一次） */
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
 * 将Base58字符串还原为字节数组（大端序）
 * 返回0成功，-1失败（非法字符或缓冲区溢出）
 */
static int base58_decode_bytes(const char *b58str, uint8_t *out, int *out_len)
{
    init_base58_decode_map();

    int cap = *out_len;
    int len = (int)strlen(b58str);

    /* 统计前导'1'（对应前导零字节） */
    int leading_ones = 0;
    while (leading_ones < len && b58str[leading_ones] == '1')
        leading_ones++;

    /* 用临时缓冲区做大数运算 */
    uint8_t tmp[128] = {0};
    int tmp_len = 0;

    for (int i = leading_ones; i < len; i++) {
        int val = base58_decode_map[(uint8_t)b58str[i]];
        if (val < 0)
            return -1; /* 非法字符 */

        int carry = val;
        for (int j = tmp_len - 1; j >= 0; j--) {
            carry += 58 * tmp[j];
            tmp[j] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
        while (carry) {
            if (tmp_len >= 128)
                return -1; /* 溢出 */
            /* 向前移一位 */
            memmove(tmp + 1, tmp, tmp_len);
            tmp[0] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
            tmp_len++;
        }
        if (tmp_len == 0 && val != 0)
            tmp_len = 1;
    }

    /* 计算总长度 */
    int total = leading_ones + tmp_len;
    if (total > cap)
        return -1; /* 缓冲区不足 */

    /* 写入输出 */
    memset(out, 0, leading_ones);
    memcpy(out + leading_ones, tmp, tmp_len);
    *out_len = total;
    return 0;
}

/* Base58Check 解码：从比特币地址字符串反向解码出20字节RIPEMD160哈希值 */
int base58check_decode(const char *b58str, uint8_t *hash160_out)
{
    uint8_t buf[64];
    int buf_len = 64;

    /* 步骤1：Base58还原为字节数组 */
    if (base58_decode_bytes(b58str, buf, &buf_len) != 0)
        return -1;

    /* 步骤2：检查长度（版本1B + hash160 20B + 校验和4B = 25字节） */
    if (buf_len != 25)
        return -1;

    /* 步骤3：校验和验证 */
    uint8_t hash[32];
    sha256d(buf, 21, hash); /* 对前21字节做双重SHA256 */
    if (memcmp(hash, buf + 21, 4) != 0)
        return -2;

    /* 步骤4：再转换为base58编码，与原始编码对比 */
    char address[64];
    base58check_encode(buf, 21, address);
    if (strcmp(b58str, address) != 0)
        return -2;

    /* 步骤5：提取hash160（跳过版本字节） */
    if (hash160_out != NULL)
        memcpy(hash160_out, buf + 1, 20);

    return 0;
}

/*
 * 直接从已序列化的公钥字节计算hash160（SHA256 -> RIPEMD160）
 */
void pubkey_bytes_to_hash160(const uint8_t *pubkey_bytes, size_t len,
                            uint8_t *hash160_out)
{
    uint8_t sha256_result[32];
    sha256(pubkey_bytes, len, sha256_result);
    ripemd160(sha256_result, 32, hash160_out);
}

/*
 * 从32字节私钥计算压缩与非压缩公钥的hash160（SHA256 -> RIPEMD160），
 * compressed_hash160   : 压缩公钥的hash160输出（20字节），传NULL则跳过
 * uncompressed_hash160 : 非压缩公钥的hash160输出（20字节），传NULL则跳过
 * 返回值：0成功，-1失败（公钥生成失败）
 */
int privkey_to_hash160(const uint8_t *privkey,
                       uint8_t *compressed_hash160,
                       uint8_t *uncompressed_hash160)
{
    uint8_t sha256_result[32];

    /* 一次椭圆曲线运算，得到公钥对象 */
    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey))
        return -1;

    /* ---- 压缩公钥 hash160 ---- */
    if (compressed_hash160 != NULL) {
        uint8_t pubkey_compressed[33];
        size_t pubkey_len1 = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_compressed, &pubkey_len1,
            &pubkey, SECP256K1_EC_COMPRESSED);
        sha256(pubkey_compressed, 33, sha256_result);
        ripemd160(sha256_result, 32, compressed_hash160);
    }

    /* ---- 非压缩公钥 hash160 ---- */
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
 * 从32字节私钥同时计算压缩与非压缩两种比特币地址（P2PKH）
 */
int privkey_to_address(const uint8_t *privkey,
                       char *compressed_out,
                       char *uncompressed_out)
{
    uint8_t hash160_compressed[20];
    uint8_t hash160_uncompressed[20];
    uint8_t versioned[21];

    /* 复用privkey_to_hash160计算两种hash160 */
    if (privkey_to_hash160(privkey, hash160_compressed, hash160_uncompressed) != 0)
        return -1;

    /* ---- 压缩地址 ---- */
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160_compressed, 20);
    base58check_encode(versioned, 21, compressed_out);

    /* ---- 非压缩地址 ---- */
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160_uncompressed, 20);
    base58check_encode(versioned, 21, uncompressed_out);

    return 0;
}

