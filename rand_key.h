#ifndef RAND_KEY_H
#define RAND_KEY_H

#include <stdint.h>
#include <stddef.h>

/* 每线程随机缓冲区大小：一次读取8KB，可供8192-32次私钥生成使用 */
#define RAND_BUF_SIZE   (8192)

typedef struct {
    uint8_t  buf[RAND_BUF_SIZE];
    size_t   pos;
    int      fd;
} rand_key_context;

/* 初始化随机上下文，打开 /dev/urandom */
int rand_ctx_init(rand_key_context *ctx);

/* 填充缓冲区 */
int rand_ctx_refill(rand_key_context *ctx);

/* 生成32字节真随机私钥 */
int gen_random_key(uint8_t *key32, rand_key_context *ctx);

#endif /* RAND_KEY_H */

