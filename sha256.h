#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ===================== SHA256 ===================== */

typedef struct {
    uint32_t state[8];
    uint32_t count[2];
    uint8_t  buf[64];
} sha256_ctx;

#define ROR32(x,n)     (((x)>>(n))|((x)<<(32-(n))))
#define SHA256_CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define SHA256_MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define SHA256_EP0(x)  (ROR32(x,2)  ^ ROR32(x,13) ^ ROR32(x,22))
#define SHA256_EP1(x)  (ROR32(x,6)  ^ ROR32(x,11) ^ ROR32(x,25))
#define SHA256_SIG0(x) (ROR32(x,7)  ^ ROR32(x,18) ^ ((x)>>3))
#define SHA256_SIG1(x) (ROR32(x,17) ^ ROR32(x,19) ^ ((x)>>10))

void sha256_init(sha256_ctx *ctx);
void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len);
void sha256_final(sha256_ctx *ctx, uint8_t *digest);
void sha256(const uint8_t *data, size_t len, uint8_t *digest);
void sha256_33(const uint8_t *data33, uint8_t *digest);   /* 压缩公钥33字节 */
void sha256_65(const uint8_t *data65, uint8_t *digest);   /* 非压缩公钥65字节 */
void sha256_compress_avx2(uint32_t *states[8], const uint8_t *blocks[8]);

#endif /* SHA256_H */

