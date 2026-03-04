#ifndef HASH_UTILS_H
#define HASH_UTILS_H

#include <stdint.h>
#include <stddef.h>

/* 私钥字节数组转十六进制字符串 */
void bytes_to_hex(const uint8_t *bytes, int len, char *hex_out);

/* SHA256两次哈希 */
void sha256d(const uint8_t *data, size_t len, uint8_t *out);

/* Base58Check编码 */
void base58check_encode(const uint8_t *payload, size_t payload_len, char *out);

/*
 * Base58Check解码：从比特币地址字符串反向解码出20字节RIPEMD160哈希值
 * b58str      : 输入的Base58Check编码字符串
 * hash160_out : 输出缓冲区，至少20字节；传入NULL时仅做校验
 * 返回值：
 *   0  : 解码成功，hash160已写入hash160_out
 *  -1  : 格式错误（含非法字符、长度不足、缓冲区溢出）
 *  -2  : 校验和不匹配
 */
int base58check_decode(const char *b58str, uint8_t *hash160_out);

/*
 * 从32字节私钥计算压缩与非压缩公钥的hash160（SHA256 -> RIPEMD160），
 * compressed_hash160   : 压缩公钥hash160输出（20字节），传NULL则跳过
 * uncompressed_hash160 : 非压缩公钥hash160输出（20字节），传NULL则跳过
 * 返回值：0成功，-1失败（公钥生成失败）
 */
int privkey_to_hash160(const uint8_t *privkey,
                       uint8_t *compressed_hash160,
                       uint8_t *uncompressed_hash160);

/*
 * 直接从已序列化的公钥字节计算hash160（SHA256 -> RIPEMD160）
 * pubkey_bytes : 已序列化的公钥字节（压缩33字节或非压缩65字节）
 * len          : 公钥字节长度（33或65）
 * hash160_out  : 输出缓冲区，至少20字节
 */
void pubkey_bytes_to_hash160(const uint8_t *pubkey_bytes, size_t len,
                              uint8_t *hash160_out);

/*
 * 从32字节私钥同时计算压缩与非压缩两种比特币地址（P2PKH）
 * compressed_out   : 压缩地址输出缓冲区（至少ADDRESS_LEN+1字节）
 * uncompressed_out : 非压缩地址输出缓冲区（至少ADDRESS_LEN+1字节）
 */
int privkey_to_address(const uint8_t *privkey,
                       char *compressed_out,
                       char *uncompressed_out);

#ifdef __AVX2__
/*
 * 8路并行计算压缩公钥（33字节）的hash160（SHA256->RIPEMD160）
 * pubkeys[8]    : 8个压缩公钥字节指针（每个33字节）
 * hash160s[8]   : 8个输出缓冲区（每个20字节）
 */
void hash160_8way_compressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20]);

/*
 * 8路并行计算非压缩公钥（65字节）的hash160（SHA256->RIPEMD160）
 * pubkeys[8]    : 8个非压缩公钥字节指针（每个65字节）
 * hash160s[8]   : 8个输出缓冲区（每个20字节）
 */
void hash160_8way_uncompressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20]);
#endif /* __AVX2__ */

/*
 * 哈希表槽位结构：
 *   fp     : hash160前4字节指纹，用于快速过滤（0表示空槽）
 *   h160   : 完整20字节hash160
 */
struct ht_slot {
    uint32_t fp;        /* 前4字节指纹（大端序读取） */
    uint8_t h160[20];  /* 完整20字节hash160 */
};

/* 全局哈希表（由 ht_init 分配，ht_free 释放） */
extern struct ht_slot *ht_slots;
extern uint32_t ht_mask;   /* 槽位掩码 = 槽位数 - 1（槽位数为2的幂次） */

/*
 * 初始化哈希表：分配capacity个槽位（capacity必须为2的幂次）
 * 返回0成功，-1失败
 */
int ht_init(uint32_t capacity);

/* 释放哈希表内存 */
void ht_free(void);

/*
 * 向哈希表插入hash160（20字节）
 * 使用线性探测解决冲突
 */
void ht_insert(const uint8_t *h160);

/*
 * 在哈希表中查找hash160（20字节）
 * 返回1命中，0未命中
 */
int ht_contains(const uint8_t *h160);

#ifdef __AVX2__
/*
 * 8路并行查表：同时查找8个hash160
 * h160s[8]: 8个hash160指针（每个20字节）
 * 返回8位命中掩码：bit i为1表示第i路命中
 */
uint8_t ht_contains_8way(const uint8_t *h160s[8]);
#endif /* __AVX2__ */

#endif /* HASH_UTILS_H */

