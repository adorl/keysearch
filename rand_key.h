#ifndef RAND_KEY_H
#define RAND_KEY_H

#include <stdint.h>
#include <stddef.h>

/* Per-thread random buffer size: read 8KB at once, sufficient for 8192/32 private key generations */
#define RAND_BUF_SIZE   (8192)

typedef struct {
    uint8_t  buf[RAND_BUF_SIZE];
    size_t   pos;
    int      fd;
} rand_key_context;

/* Initialize random context, open /dev/urandom */
int rand_ctx_init(rand_key_context *ctx);

/* Refill buffer */
int rand_ctx_refill(rand_key_context *ctx);

/* Generate 32-byte true random private key */
int gen_random_key(uint8_t *key32, rand_key_context *ctx);

#endif /* RAND_KEY_H */

