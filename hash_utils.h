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

#endif /* HASH_UTILS_H */

