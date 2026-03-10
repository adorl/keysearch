#include "sha256.h"

/* Fully unrolled SHA256 compression function: eliminates loop/array access, lets compiler allocate variables to registers */
#define SHA256_LOAD(i) \
    (((uint32_t)block[(i) * 4] << 24) | ((uint32_t)block[(i) * 4 + 1] << 16) | ((uint32_t)block[(i) * 4 + 2] << 8) | (uint32_t)block[(i) * 4 + 3])

#define SHA256_EXPAND(w0, w1, w9, w14) \
    (SHA256_SIG1(w14) + (w9) + SHA256_SIG0(w1) + (w0))

/* One SHA256 round: a..h -> new a..h, k is constant, w is message word */
#define SHA256_ROUND(a, b, c, d, e, f, g, h, k, w)                              \
    do {                                                                        \
        uint32_t _t1 = (h) + SHA256_EP1(e) + SHA256_CH(e, f, g) + (k) + (w);    \
        uint32_t _t2 = SHA256_EP0(a) + SHA256_MAJ(a, b, c);                     \
        (d) += _t1;                                                             \
        (h) = _t1 + _t2;                                                        \
    } while (0)

static void sha256_compress(uint32_t *state, const uint8_t *block)
{
    /* Load message block (big-endian) */
    uint32_t w0 = SHA256_LOAD(0), w1 = SHA256_LOAD(1);
    uint32_t w2 = SHA256_LOAD(2), w3 = SHA256_LOAD(3);
    uint32_t w4 = SHA256_LOAD(4), w5 = SHA256_LOAD(5);
    uint32_t w6 = SHA256_LOAD(6), w7 = SHA256_LOAD(7);
    uint32_t w8 = SHA256_LOAD(8), w9 = SHA256_LOAD(9);
    uint32_t w10 = SHA256_LOAD(10), w11 = SHA256_LOAD(11);
    uint32_t w12 = SHA256_LOAD(12), w13 = SHA256_LOAD(13);
    uint32_t w14 = SHA256_LOAD(14), w15 = SHA256_LOAD(15);

    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];

    /* Rounds 0-15 (use message words directly) */
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x428a2f98, w0);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x71374491, w1);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb5c0fbcf, w2);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xe9b5dba5, w3);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x3956c25b, w4);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x59f111f1, w5);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x923f82a4, w6);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xab1c5ed5, w7);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xd807aa98, w8);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x12835b01, w9);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x243185be, w10);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x550c7dc3, w11);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x72be5d74, w12);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x80deb1fe, w13);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x9bdc06a7, w14);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc19bf174, w15);

    /* Rounds 16-31 (use after message expansion) */
    w0 = SHA256_EXPAND(w0, w1, w9, w14);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xe49b69c1, w0);
    w1 = SHA256_EXPAND(w1, w2, w10, w15);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xefbe4786, w1);
    w2 = SHA256_EXPAND(w2, w3, w11, w0);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x0fc19dc6, w2);
    w3 = SHA256_EXPAND(w3, w4, w12, w1);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x240ca1cc, w3);
    w4 = SHA256_EXPAND(w4, w5, w13, w2);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x2de92c6f, w4);
    w5 = SHA256_EXPAND(w5, w6, w14, w3);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4a7484aa, w5);
    w6 = SHA256_EXPAND(w6, w7, w15, w4);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5cb0a9dc, w6);
    w7 = SHA256_EXPAND(w7, w8, w0, w5);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x76f988da, w7);
    w8 = SHA256_EXPAND(w8, w9, w1, w6);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x983e5152, w8);
    w9 = SHA256_EXPAND(w9, w10, w2, w7);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa831c66d, w9);
    w10 = SHA256_EXPAND(w10, w11, w3, w8);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xb00327c8, w10);
    w11 = SHA256_EXPAND(w11, w12, w4, w9);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xbf597fc7, w11);
    w12 = SHA256_EXPAND(w12, w13, w5, w10);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0xc6e00bf3, w12);
    w13 = SHA256_EXPAND(w13, w14, w6, w11);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd5a79147, w13);
    w14 = SHA256_EXPAND(w14, w15, w7, w12);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x06ca6351, w14);
    w15 = SHA256_EXPAND(w15, w0, w8, w13);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x14292967, w15);

    /* Rounds 32-47 */
    w0 = SHA256_EXPAND(w0, w1, w9, w14);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x27b70a85, w0);
    w1 = SHA256_EXPAND(w1, w2, w10, w15);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x2e1b2138, w1);
    w2 = SHA256_EXPAND(w2, w3, w11, w0);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x4d2c6dfc, w2);
    w3 = SHA256_EXPAND(w3, w4, w12, w1);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x53380d13, w3);
    w4 = SHA256_EXPAND(w4, w5, w13, w2);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x650a7354, w4);
    w5 = SHA256_EXPAND(w5, w6, w14, w3);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x766a0abb, w5);
    w6 = SHA256_EXPAND(w6, w7, w15, w4);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x81c2c92e, w6);
    w7 = SHA256_EXPAND(w7, w8, w0, w5);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x92722c85, w7);
    w8 = SHA256_EXPAND(w8, w9, w1, w6);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0xa2bfe8a1, w8);
    w9 = SHA256_EXPAND(w9, w10, w2, w7);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0xa81a664b, w9);
    w10 = SHA256_EXPAND(w10, w11, w3, w8);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0xc24b8b70, w10);
    w11 = SHA256_EXPAND(w11, w12, w4, w9);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0xc76c51a3, w11);
    w12 = SHA256_EXPAND(w12, w13, w5, w10);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0xd192e819, w12);
    w13 = SHA256_EXPAND(w13, w14, w6, w11);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xd6990624, w13);
    w14 = SHA256_EXPAND(w14, w15, w7, w12);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0xf40e3585, w14);
    w15 = SHA256_EXPAND(w15, w0, w8, w13);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x106aa070, w15);

    /* Rounds 48-63 */
    w0 = SHA256_EXPAND(w0, w1, w9, w14);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x19a4c116, w0);
    w1 = SHA256_EXPAND(w1, w2, w10, w15);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x1e376c08, w1);
    w2 = SHA256_EXPAND(w2, w3, w11, w0);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x2748774c, w2);
    w3 = SHA256_EXPAND(w3, w4, w12, w1);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x34b0bcb5, w3);
    w4 = SHA256_EXPAND(w4, w5, w13, w2);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x391c0cb3, w4);
    w5 = SHA256_EXPAND(w5, w6, w14, w3);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0x4ed8aa4a, w5);
    w6 = SHA256_EXPAND(w6, w7, w15, w4);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0x5b9cca4f, w6);
    w7 = SHA256_EXPAND(w7, w8, w0, w5);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0x682e6ff3, w7);
    w8 = SHA256_EXPAND(w8, w9, w1, w6);
    SHA256_ROUND(a, b, c, d, e, f, g, h, 0x748f82ee, w8);
    w9 = SHA256_EXPAND(w9, w10, w2, w7);
    SHA256_ROUND(h, a, b, c, d, e, f, g, 0x78a5636f, w9);
    w10 = SHA256_EXPAND(w10, w11, w3, w8);
    SHA256_ROUND(g, h, a, b, c, d, e, f, 0x84c87814, w10);
    w11 = SHA256_EXPAND(w11, w12, w4, w9);
    SHA256_ROUND(f, g, h, a, b, c, d, e, 0x8cc70208, w11);
    w12 = SHA256_EXPAND(w12, w13, w5, w10);
    SHA256_ROUND(e, f, g, h, a, b, c, d, 0x90befffa, w12);
    w13 = SHA256_EXPAND(w13, w14, w6, w11);
    SHA256_ROUND(d, e, f, g, h, a, b, c, 0xa4506ceb, w13);
    w14 = SHA256_EXPAND(w14, w15, w7, w12);
    SHA256_ROUND(c, d, e, f, g, h, a, b, 0xbef9a3f7, w14);
    w15 = SHA256_EXPAND(w15, w0, w8, w13);
    SHA256_ROUND(b, c, d, e, f, g, h, a, 0xc67178f2, w15);

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
 * Specialized SHA256 for fixed 33-byte input (compressed pubkey), single-block processing
 * 33 bytes + 0x80 + 22 zero bytes + 8 bytes length = 64 bytes
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
    /* Length: 33*8=264 bits=0x108, written big-endian into last 8 bytes */
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
 * Specialized SHA256 for fixed 65-byte input (uncompressed pubkey), two-block processing.
 * First block: first 64 bytes of 65 bytes
 * Second block: 65th byte + 0x80 + 54 zero bytes + 8 bytes length
 */
void sha256_65(const uint8_t *data65, uint8_t *digest)
{
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    /* First block: use first 64 bytes directly */
    sha256_compress(state, data65);

    /* Second block: 65th byte + padding + length */
    uint8_t block2[64];
    block2[0] = data65[64];
    block2[1] = 0x80;
    memset(block2 + 2, 0, 64 - 2 - 8);
    /* Length: 65*8=520 bits=0x208, big-endian */
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

