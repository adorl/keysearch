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

struct ht_slot *ht_slots = NULL;
uint32_t ht_mask = 0;

/* 初始化哈希表：分配capacity个槽位（capacity必须为2的幂次） */
int ht_init(uint32_t capacity)
{
    ht_slots = (struct ht_slot *)calloc(capacity, sizeof(struct ht_slot));
    if (!ht_slots)
        return -1;
    ht_mask = capacity - 1;
    return 0;
}

/* 释放哈希表内存 */
void ht_free(void)
{
    free(ht_slots);
    ht_slots = NULL;
    ht_mask = 0;
}

/*
 * FNV-1a 哈希：将hash160前4字节读为大端uint32_t，再做FNV-1a乘法哈希
 * 返回槽位索引（已 & ht_mask）
 */
static inline uint32_t ht_hash(const uint8_t *h160)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < 20; i++) {
        h ^= h160[i];
        h *= 16777619u;
    }
    return h & ht_mask;
}

/* 向哈希表插入hash160（线性探测） */
void ht_insert(const uint8_t *h160)
{
    /* 提取前4字节指纹（大端序） */
    uint32_t fp = ((uint32_t)h160[0] << 24) |
                  ((uint32_t)h160[1] << 16) |
                  ((uint32_t)h160[2] <<  8) |
                   (uint32_t)h160[3];
    /* fp == 0时用1代替，避免与空槽标识冲突 */
    if (fp == 0)
        fp = 1;

    uint32_t idx = ht_hash(h160);
    while (ht_slots[idx].fp != 0) {
        idx = (idx + 1) & ht_mask;
    }
    ht_slots[idx].fp = fp;
    memcpy(ht_slots[idx].h160, h160, 20);
}

/* 在哈希表中查找hash160（线性探测） */
int ht_contains(const uint8_t *h160)
{
    uint32_t fp = ((uint32_t)h160[0] << 24) |
                  ((uint32_t)h160[1] << 16) |
                  ((uint32_t)h160[2] <<  8) |
                   (uint32_t)h160[3];
    if (fp == 0)
        fp = 1;

    uint32_t idx = ht_hash(h160);
    while (1) {
        uint32_t slot_fp = ht_slots[idx].fp;
        if (slot_fp == 0)
            return 0; /* 空槽，未命中 */
        if (slot_fp == fp && memcmp(ht_slots[idx].h160, h160, 20) == 0)
            return 1; /* 命中 */
        idx = (idx + 1) & ht_mask;
    }
}

#ifdef __AVX2__

/* SHA256初始状态常量 */
static const uint32_t sha256_init_state[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* RIPEMD160初始状态常量 */
static const uint32_t rmd160_init_state[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/*
 * 构造SHA256 padded block（单块，消息 <= 55 字节）
 * 消息+0x80+零填充+8字节大端消息位长
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
 * 构造SHA256第一个block（64字节消息块，无padding）
 * 用于65字节消息的第一块（取前64字节）
 */
static void make_sha256_block_raw(const uint8_t *msg64, uint8_t block[64])
{
    memcpy(block, msg64, 64);
}

/*
 * 构造SHA256第二个padded block（65字节消息的尾块）
 * 消息剩余1字节+0x80+零填充+8字节大端消息位长(65*8=520bits)
 */
static void make_sha256_block2_65(const uint8_t last_byte, uint8_t block[64])
{
    memset(block, 0, 64);
    block[0] = last_byte;
    block[1] = 0x80;
    /* 消息位长 = 65 * 8 = 520 = 0x208 */
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
 * 将SHA256 state[8]转为32字节大端序摘要
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
 * 构造RIPEMD160 padded block（单块，消息 <= 55字节）
 * 消息+0x80+零填充+8字节小端消息位长
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
 * 将RIPEMD160 state[5]转为20字节小端序摘要
 */
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
 * 8路并行计算压缩公钥（33字节）的hash160（SHA256->RIPEMD160）
 *
 * 压缩公钥33字节<56字节，单块SHA256即可完成
 * SHA256输出32字节<56字节，单块RIPEMD160即可完成
 */
__attribute__((target("avx2")))
void hash160_8way_compressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20])
{
    /* ---- 步骤1：构造8路SHA256 padded block（33字节消息） ---- */
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

    /* ---- 步骤2：8路并行SHA256压缩 ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- 步骤3：提取8路SHA256摘要（32字节大端序） ---- */
    uint8_t sha_digests[8][32];
    for (int i = 0; i < 8; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
    }

    /* ---- 步骤4：构造8路RIPEMD160 padded block（32字节消息） ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block(sha_digests[i], 32, rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- 步骤5：8路并行RIPEMD160压缩 ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- 步骤6：提取8路RIPEMD160摘要（20字节小端序） ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8路并行计算非压缩公钥（65字节）的hash160（SHA256 -> RIPEMD160）
 *
 * 非压缩公钥65字节需要2个SHA256 block：
 *   block1：前64字节（原始数据，无 padding）
 *   block2：第65字节+0x80+零填充+8字节大端位长(520bits)
 */
__attribute__((target("avx2")))
void hash160_8way_uncompressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20])
{
    /* ---- 步骤1：构造8路SHA256第一个block（前64字节） ---- */
    uint8_t sha_blocks1[8][64];
    uint32_t sha_states[8][8];
    uint32_t *sha_state_ptrs[8];
    const uint8_t *sha_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_sha256_block_raw(pubkeys[i], sha_blocks1[i]);
        memcpy(sha_states[i], sha256_init_state, 32);
        sha_state_ptrs[i] = sha_states[i];
        sha_block_ptrs[i] = sha_blocks1[i];
    }

    /* ---- 步骤2：8路并行SHA256第一次压缩 ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- 步骤3：构造8路SHA256第二个padded block（第65字节 + padding） ---- */
    uint8_t sha_blocks2[8][64];
    for (int i = 0; i < 8; i++) {
        make_sha256_block2_65(pubkeys[i][64], sha_blocks2[i]);
        sha_block_ptrs[i] = sha_blocks2[i];
    }

    /* ---- 步骤4：8路并行SHA256第二次压缩（state已更新，继续累加） ---- */
    sha256_compress_avx2(sha_state_ptrs, sha_block_ptrs);

    /* ---- 步骤5：提取8路SHA256摘要（32字节大端序） ---- */
    uint8_t sha_digests[8][32];
    for (int i = 0; i < 8; i++) {
        sha256_state_to_bytes(sha_states[i], sha_digests[i]);
    }

    /* ---- 步骤6：构造8路RIPEMD160 padded block（32字节消息） ---- */
    uint8_t rmd_blocks[8][64];
    uint32_t rmd_states[8][5];
    uint32_t *rmd_state_ptrs[8];
    const uint8_t *rmd_block_ptrs[8];

    for (int i = 0; i < 8; i++) {
        make_rmd160_block(sha_digests[i], 32, rmd_blocks[i]);
        memcpy(rmd_states[i], rmd160_init_state, 20);
        rmd_state_ptrs[i] = rmd_states[i];
        rmd_block_ptrs[i] = rmd_blocks[i];
    }

    /* ---- 步骤7：8路并行RIPEMD160压缩 ---- */
    ripemd160_compress_avx2(rmd_state_ptrs, rmd_block_ptrs);

    /* ---- 步骤8：提取8路RIPEMD160摘要（20字节小端序） ---- */
    for (int i = 0; i < 8; i++) {
        rmd160_state_to_bytes(rmd_states[i], hash160s[i]);
    }
}

/*
 * 8路并行查表：同时查找8个hash160
 *
 * 策略：
 *   1. 用标量FNV-1a计算8路槽位索引（与ht_contains完全一致）
 *   2. 用AVX2 _mm256_set_epi32加载8路指纹，_mm256_cmpeq_epi32同时比较
 *   3. 对指纹命中的lane执行标量线性探测 + memcmp 20字节确认
 *   4. 返回8位命中掩码（bit i为1表示第i路命中）
 */
__attribute__((target("avx2")))
uint8_t ht_contains_8way(const uint8_t *h160s[8])
{
    /* ---- 步骤1：计算8路指纹和初始槽位索引 ---- */
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

        /* FNV-1a 哈希（与ht_hash完全一致） */
        uint32_t hv = 2166136261u;
        for (int j = 0; j < 20; j++) {
            hv ^= h[j];
            hv *= 16777619u;
        }
        idxs[i] = hv & ht_mask;
    }

    /* ---- 步骤2：AVX2同时加载8路槽位指纹并比较 ---- */

    /* 加载8路当前槽位的指纹 */
    __m256i vslot = _mm256_set_epi32(
        (int32_t)ht_slots[idxs[7]].fp, (int32_t)ht_slots[idxs[6]].fp,
        (int32_t)ht_slots[idxs[5]].fp, (int32_t)ht_slots[idxs[4]].fp,
        (int32_t)ht_slots[idxs[3]].fp, (int32_t)ht_slots[idxs[2]].fp,
        (int32_t)ht_slots[idxs[1]].fp, (int32_t)ht_slots[idxs[0]].fp);

    /* 检测空槽（fp == 0表示空槽，直接未命中） */
    __m256i vzero = _mm256_setzero_si256();
    __m256i vempty = _mm256_cmpeq_epi32(vslot, vzero); /* 空槽掩码 */

    int empty_mask = _mm256_movemask_epi8(vempty);

    /* ---- 步骤3：对指纹命中的lane执行标量线性探测+memcmp确认 ---- */
    uint8_t result = 0;

    for (int i = 0; i < 8; i++) {
        /* 每个lane对应movemask中的4个连续bit（epi32 -> 4字节） */
        int lane_bit = 1 << (i * 4);

        if (empty_mask & lane_bit) {
            /* 初始槽为空，直接未命中 */
            continue;
        }

        /* 标量线性探测：从初始槽开始，直到找到匹配或空槽 */
        uint32_t idx = idxs[i];
        uint32_t fp  = fps[i];
        const uint8_t *h = h160s[i];

        while (1) {
            uint32_t slot_fp = ht_slots[idx].fp;
            if (slot_fp == 0)
                break; /* 空槽，未命中 */
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

