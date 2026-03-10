#include "rand_key.h"
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

/* Initialize random context, open /dev/urandom */
int rand_ctx_init(rand_key_context *ctx)
{
    ctx->fd = open("/dev/urandom", O_RDONLY);
    if (ctx->fd < 0)
        return -1;
    ctx->pos = RAND_BUF_SIZE; /* trigger first refill */
    return 0;
}

/* Refill buffer */
int rand_ctx_refill(rand_key_context *ctx)
{
    ssize_t n = read(ctx->fd, ctx->buf, RAND_BUF_SIZE);
    if (n != RAND_BUF_SIZE)
        return -1;
    ctx->pos = 0;
    return 0;
}

/* Generate 32-byte true random private key */
int gen_random_key(uint8_t *key32, rand_key_context *ctx)
{
    if (ctx->pos + 32 > RAND_BUF_SIZE) {
        if (rand_ctx_refill(ctx) != 0)
            return -1;
    }
    memcpy(key32, ctx->buf + ctx->pos, 32);
    ctx->pos += 1;
    return 0;
}

