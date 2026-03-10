#ifndef RIPEMD160_H
#define RIPEMD160_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint32_t state[5];
    uint32_t count[2];
    uint8_t  buf[64];
} ripemd160_ctx;

#define RMD_F(x,y,z)  ((x) ^ (y) ^ (z))
#define RMD_G(x,y,z)  (((x) & (y)) | (~(x) & (z)))
#define RMD_H(x,y,z)  (((x) | ~(y)) ^ (z))
#define RMD_I(x,y,z)  (((x) & (z)) | ((y) & ~(z)))
#define RMD_J(x,y,z)  ((x) ^ ((y) | ~(z)))
#define ROL32(x,n)    (((x) << (n)) | ((x) >> (32-(n))))

void ripemd160_init(ripemd160_ctx *ctx);
void ripemd160_update(ripemd160_ctx *ctx, const uint8_t *data, size_t len);
void ripemd160_final(ripemd160_ctx *ctx, uint8_t *digest);
void ripemd160(const uint8_t *data, size_t len, uint8_t *digest);
void ripemd160_32(const uint8_t *data32, uint8_t *digest);
void ripemd160_compress_avx2(uint32_t *states[8], const uint8_t *blocks[8]);

#endif /* RIPEMD160_H */

