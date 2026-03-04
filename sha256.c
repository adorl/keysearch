#include "sha256.h"

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_compress(uint32_t *state, const uint8_t *block)
{
    uint32_t w[64], a, b, c, d, e, f, g, h, t1, t2;
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)
              |((uint32_t)block[i*4+2]<<8)|(uint32_t)block[i*4+3];
    for (int i = 16; i < 64; i++)
        w[i] = SHA256_SIG1(w[i-2]) + w[i-7] + SHA256_SIG0(w[i-15]) + w[i-16];
    a=state[0]; b=state[1]; c=state[2]; d=state[3];
    e=state[4]; f=state[5]; g=state[6]; h=state[7];
    for (int i = 0; i < 64; i++) {
        t1 = h + SHA256_EP1(e) + SHA256_CH(e,f,g) + SHA256_K[i] + w[i];
        t2 = SHA256_EP0(a) + SHA256_MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1;
        d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

void sha256_init(sha256_ctx *ctx)
{
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
    ctx->count[0] = ctx->count[1] = 0;
}

void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len)
{
    size_t have = (ctx->count[0] >> 3) & 63;
    if ((ctx->count[0] += (uint32_t)(len << 3)) < (uint32_t)(len << 3))
        ctx->count[1]++;
    ctx->count[1] += (uint32_t)(len >> 29);
    size_t need = 64 - have, off = 0;
    if (len >= need) {
        memcpy(ctx->buf + have, data, need);
        sha256_compress(ctx->state, ctx->buf);
        off = need; have = 0;
        while (off + 64 <= len) {
            sha256_compress(ctx->state, data + off);
            off += 64;
        }
    }
    memcpy(ctx->buf + have, data + off, len - off);
}

void sha256_final(sha256_ctx *ctx, uint8_t *digest)
{
    uint8_t pad[64] = {0x80};
    uint8_t len_buf[8];
    for (int i = 0; i < 4; i++) {
        len_buf[i]   = (ctx->count[1] >> (24 - i*8)) & 0xFF;
        len_buf[i+4] = (ctx->count[0] >> (24 - i*8)) & 0xFF;
    }
    size_t have = (ctx->count[0] >> 3) & 63;
    size_t pad_len = (have < 56) ? (56 - have) : (120 - have);
    sha256_update(ctx, pad, pad_len);
    sha256_update(ctx, len_buf, 8);
    for (int i = 0; i < 8; i++) {
        digest[i*4]   = (ctx->state[i] >> 24) & 0xFF;
        digest[i*4+1] = (ctx->state[i] >> 16) & 0xFF;
        digest[i*4+2] = (ctx->state[i] >>  8) & 0xFF;
        digest[i*4+3] = (ctx->state[i])        & 0xFF;
    }
}

void sha256(const uint8_t *data, size_t len, uint8_t *digest)
{
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}

/*
 * 针对固定33字节输入（压缩公钥）的专用SHA256，单块处理
 * 33字节+0x80+22字节零+8字节长度=64字节
 */
void sha256_33(const uint8_t *data33, uint8_t *digest)
{
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint8_t block[64];
    memcpy(block, data33, 33);
    block[33] = 0x80;
    memset(block + 34, 0, 64 - 34 - 8);
    /* 长度：33*8=264bits=0x108，大端序写入最后8字节 */
    block[56] = 0x00;
    block[57] = 0x00;
    block[58] = 0x00;
    block[59] = 0x00;
    block[60] = 0x00;
    block[61] = 0x00;
    block[62] = 0x01;
    block[63] = 0x08;

    sha256_compress(state, block);

    for (int i = 0; i < 8; i++) {
        digest[i * 4] = (state[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = (state[i]) & 0xFF;
    }
}

/*
 * 针对固定65字节输入（非压缩公钥）的专用SHA256，两块处理。
 * 第一块：65字节中的前64字节
 * 第二块：第65字节+0x80+54字节零+8字节长度
 */
void sha256_65(const uint8_t *data65, uint8_t *digest)
{
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    /* 第一块：直接使用前64字节 */
    sha256_compress(state, data65);

    /* 第二块：第65字节 + padding + 长度 */
    uint8_t block2[64];
    block2[0] = data65[64];
    block2[1] = 0x80;
    memset(block2 + 2, 0, 64 - 2 - 8);
    /* 长度：65*8=520bits=0x208，大端序 */
    block2[56] = 0x00;
    block2[57] = 0x00;
    block2[58] = 0x00;
    block2[59] = 0x00;
    block2[60] = 0x00;
    block2[61] = 0x00;
    block2[62] = 0x02;
    block2[63] = 0x08;

    sha256_compress(state, block2);

    for (int i = 0; i < 8; i++) {
        digest[i * 4] = (state[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = (state[i]) & 0xFF;
    }
}

