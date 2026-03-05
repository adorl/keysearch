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
 * 8路并行计算压缩公钥的hash160（预填充，零拷贝）
 * blocks[8]     : 8个已完成SHA256padding的64字节block指针
 * hash160s[8]   : 8个输出缓冲区（每个20字节）
 */
void hash160_8way_compressed_prepadded(const uint8_t *blocks[8], uint8_t hash160s[8][20]);

/*
 * 将33字节压缩公钥原地构造为SHA256 padded block（64字节）
 * buf必须至少64字节，公钥数据已写入buf[0..32]，本函数补全padding
 */
static inline void sha256_pad_block_33(uint8_t buf[64])
{
    /* 消息长度33字节，0x80 padding，位长 = 33*8 = 264 = 0x108 */
    buf[33] = 0x80;
    buf[34] = 0x00; buf[35] = 0x00; buf[36] = 0x00; buf[37] = 0x00;
    buf[38] = 0x00; buf[39] = 0x00; buf[40] = 0x00; buf[41] = 0x00;
    buf[42] = 0x00; buf[43] = 0x00; buf[44] = 0x00; buf[45] = 0x00;
    buf[46] = 0x00; buf[47] = 0x00; buf[48] = 0x00; buf[49] = 0x00;
    buf[50] = 0x00; buf[51] = 0x00; buf[52] = 0x00; buf[53] = 0x00;
    buf[54] = 0x00; buf[55] = 0x00;
    /* 大端序位长264 = 0x0000000000000108 */
    buf[56] = 0x00; buf[57] = 0x00; buf[58] = 0x00; buf[59] = 0x00;
    buf[60] = 0x00; buf[61] = 0x00; buf[62] = 0x01; buf[63] = 0x08;
}

/*
 * 8路并行计算非压缩公钥（65字节）的hash160（SHA256->RIPEMD160）
 * pubkeys[8]    : 8个非压缩公钥字节指针（每个65字节）
 * hash160s[8]   : 8个输出缓冲区（每个20字节）
 */
void hash160_8way_uncompressed(const uint8_t *pubkeys[8], uint8_t hash160s[8][20]);

/*
 * 8路并行计算非压缩公钥的hash160（预填充，零拷贝）
 * bufs[8]       : 8个128字节buffer指针，布局如下：
 *                   buf[0..63] = 公钥前64字节
 *                   buf[64..127]= SHA256 block2
 * hash160s[8]   : 8个输出缓冲区（每个20字节）
 */
void hash160_8way_uncompressed_prepadded(const uint8_t *bufs[8], uint8_t hash160s[8][20]);

/*
 * 对65字节非压缩公钥buffer原地构造SHA256 block2（写入buf[64..127]）
 * buf必须至少128字节，公钥数据已写入buf[0..64]，本函数在buf[64..127]补全block2
 * 调用前：buf[0..64] = 非压缩公钥（65字节）
 * 调用后：buf[64..127] = SHA256 block2
 */
static inline void sha256_pad_block2_65(uint8_t buf[128])
{
    /* 读取公钥第65字节（buf[64]），构造block2 */
    uint8_t last_byte = buf[64];
    /* block2：last_byte + 0x80 + 零填充 + 8字节大端位长(65*8=520=0x208) */
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
    /* 大端序位长 520 = 0x0000000000000208 */
    buf[120] = 0x00; buf[121] = 0x00; buf[122] = 0x00; buf[123] = 0x00;
    buf[124] = 0x00; buf[125] = 0x00; buf[126] = 0x02; buf[127] = 0x08;
}
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

#ifdef __AVX512F__
/*
 * 16路并行计算压缩公钥（33字节）的hash160（SHA256->RIPEMD160）
 * pubkeys[16]   : 16个压缩公钥字节指针（每个33字节）
 * hash160s[16]  : 16个输出缓冲区（每个20字节）
 */
void hash160_16way_compressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20]);

/*
 * 16路并行计算非压缩公钥（65字节）的hash160（SHA256->RIPEMD160）
 * pubkeys[16]   : 16个非压缩公钥字节指针（每个65字节）
 * hash160s[16]  : 16个输出缓冲区（每个20字节）
 */
void hash160_16way_uncompressed(const uint8_t *pubkeys[16], uint8_t hash160s[16][20]);

/*
 * 16路并行查表：同时查找16个hash160
 * h160s[16]: 16个hash160指针（每个20字节）
 * 返回16位命中掩码：bit i为1表示第i路命中
 */
uint16_t ht_contains_16way(const uint8_t *h160s[16]);
#endif /* __AVX512F__ */

#endif /* HASH_UTILS_H */

