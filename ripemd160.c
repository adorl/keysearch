#include "ripemd160.h"

/* Fully unrolled RIPEMD160 compression function: eliminates loop/array access */
/* Left chain step macro: F(b,c,d)=b^c^d */
#define RL_F(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + (b ^ c ^ d) + (x) + 0x00000000, (s)) + (e);   \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
/* Left chain step macro: G(b,c,d)=(b&c)|(~b&d) */
#define RL_G(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b & c) | (~(b) & d)) + (x) + 0x5A827999, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
/* Left chain step macro: H(b,c,d)=(b|~c)^d */
#define RL_H(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b | ~(c)) ^ d) + (x) + 0x6ED9EBA1, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
/* Left chain step macro: I(b,c,d)=(b&d)|(c&~d) */
#define RL_I(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b & d) | (c & ~(d))) + (x) + 0x8F1BBCDC, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
/* Left chain step macro: J(b,c,d)=b^(c|~d) */
#define RL_J(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b) ^ (c | ~(d))) + (x) + 0xA953FD4E, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
/* Right chain step macros */
#define RR_J(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b) ^ (c | ~(d))) + (x) + 0x50A28BE6, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
#define RR_I(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b & d) | (c & ~(d))) + (x) + 0x5C4DD124, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
#define RR_H(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b | ~(c)) ^ d) + (x) + 0x6D703EF3, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
#define RR_G(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + ((b & c) | (~(b) & d)) + (x) + 0x7A6D76E9, (s)) + (e); \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)
#define RR_F(a, b, c, d, e, x, s)                                               \
    do {                                                                        \
        uint32_t _t = ROL32((a) + (b ^ c ^ d) + (x) + 0x00000000, (s)) + (e);   \
        (a) = (e);                                                              \
        (e) = (d);                                                              \
        (d) = ROL32((c), 10);                                                   \
        (c) = (b);                                                              \
        (b) = _t;                                                               \
    } while (0)

static void ripemd160_compress(uint32_t *state, const uint8_t *block)
{
    /* Load message words (little-endian) */
#define W(i) ((uint32_t)block[(i) * 4] | ((uint32_t)block[(i) * 4 + 1] << 8) | ((uint32_t)block[(i) * 4 + 2] << 16) | ((uint32_t)block[(i) * 4 + 3] << 24))
    uint32_t w0 = W(0), w1 = W(1), w2 = W(2), w3 = W(3), w4 = W(4), w5 = W(5), w6 = W(6), w7 = W(7);
    uint32_t w8 = W(8), w9 = W(9), w10 = W(10), w11 = W(11), w12 = W(12), w13 = W(13), w14 = W(14), w15 = W(15);
#undef W

    uint32_t al = state[0], bl = state[1], cl = state[2], dl = state[3], el = state[4];
    uint32_t ar = state[0], br = state[1], cr = state[2], dr = state[3], er = state[4];

    /* Left chain: rounds 0-15, F(b,c,d) = b^c^d */
    RL_F(al, bl, cl, dl, el, w0, 11);
    RL_F(al, bl, cl, dl, el, w1, 14);
    RL_F(al, bl, cl, dl, el, w2, 15);
    RL_F(al, bl, cl, dl, el, w3, 12);
    RL_F(al, bl, cl, dl, el, w4, 5);
    RL_F(al, bl, cl, dl, el, w5, 8);
    RL_F(al, bl, cl, dl, el, w6, 7);
    RL_F(al, bl, cl, dl, el, w7, 9);
    RL_F(al, bl, cl, dl, el, w8, 11);
    RL_F(al, bl, cl, dl, el, w9, 13);
    RL_F(al, bl, cl, dl, el, w10, 14);
    RL_F(al, bl, cl, dl, el, w11, 15);
    RL_F(al, bl, cl, dl, el, w12, 6);
    RL_F(al, bl, cl, dl, el, w13, 7);
    RL_F(al, bl, cl, dl, el, w14, 9);
    RL_F(al, bl, cl, dl, el, w15, 8);
    /* Left chain: rounds 16-31, G(b,c,d) = (b&c)|(~b&d) */
    RL_G(al, bl, cl, dl, el, w7, 7);
    RL_G(al, bl, cl, dl, el, w4, 6);
    RL_G(al, bl, cl, dl, el, w13, 8);
    RL_G(al, bl, cl, dl, el, w1, 13);
    RL_G(al, bl, cl, dl, el, w10, 11);
    RL_G(al, bl, cl, dl, el, w6, 9);
    RL_G(al, bl, cl, dl, el, w15, 7);
    RL_G(al, bl, cl, dl, el, w3, 15);
    RL_G(al, bl, cl, dl, el, w12, 7);
    RL_G(al, bl, cl, dl, el, w0, 12);
    RL_G(al, bl, cl, dl, el, w9, 15);
    RL_G(al, bl, cl, dl, el, w5, 9);
    RL_G(al, bl, cl, dl, el, w2, 11);
    RL_G(al, bl, cl, dl, el, w14, 7);
    RL_G(al, bl, cl, dl, el, w11, 13);
    RL_G(al, bl, cl, dl, el, w8, 12);
    /* Left chain: rounds 32-47, H(b,c,d) = (b|~c)^d */
    RL_H(al, bl, cl, dl, el, w3, 11);
    RL_H(al, bl, cl, dl, el, w10, 13);
    RL_H(al, bl, cl, dl, el, w14, 6);
    RL_H(al, bl, cl, dl, el, w4, 7);
    RL_H(al, bl, cl, dl, el, w9, 14);
    RL_H(al, bl, cl, dl, el, w15, 9);
    RL_H(al, bl, cl, dl, el, w8, 13);
    RL_H(al, bl, cl, dl, el, w1, 15);
    RL_H(al, bl, cl, dl, el, w2, 14);
    RL_H(al, bl, cl, dl, el, w7, 8);
    RL_H(al, bl, cl, dl, el, w0, 13);
    RL_H(al, bl, cl, dl, el, w6, 6);
    RL_H(al, bl, cl, dl, el, w13, 5);
    RL_H(al, bl, cl, dl, el, w11, 12);
    RL_H(al, bl, cl, dl, el, w5, 7);
    RL_H(al, bl, cl, dl, el, w12, 5);
    /* Left chain: rounds 48-63, I(b,c,d) = (b&d)|(c&~d) */
    RL_I(al, bl, cl, dl, el, w1, 11);
    RL_I(al, bl, cl, dl, el, w9, 12);
    RL_I(al, bl, cl, dl, el, w11, 14);
    RL_I(al, bl, cl, dl, el, w10, 15);
    RL_I(al, bl, cl, dl, el, w0, 14);
    RL_I(al, bl, cl, dl, el, w8, 15);
    RL_I(al, bl, cl, dl, el, w12, 9);
    RL_I(al, bl, cl, dl, el, w4, 8);
    RL_I(al, bl, cl, dl, el, w13, 9);
    RL_I(al, bl, cl, dl, el, w3, 14);
    RL_I(al, bl, cl, dl, el, w7, 5);
    RL_I(al, bl, cl, dl, el, w15, 6);
    RL_I(al, bl, cl, dl, el, w14, 8);
    RL_I(al, bl, cl, dl, el, w5, 6);
    RL_I(al, bl, cl, dl, el, w6, 5);
    RL_I(al, bl, cl, dl, el, w2, 12);
    /* Left chain: rounds 64-79, J(b,c,d) = b^(c|~d) */
    RL_J(al, bl, cl, dl, el, w4, 9);
    RL_J(al, bl, cl, dl, el, w0, 15);
    RL_J(al, bl, cl, dl, el, w5, 5);
    RL_J(al, bl, cl, dl, el, w9, 11);
    RL_J(al, bl, cl, dl, el, w7, 6);
    RL_J(al, bl, cl, dl, el, w12, 8);
    RL_J(al, bl, cl, dl, el, w2, 13);
    RL_J(al, bl, cl, dl, el, w10, 12);
    RL_J(al, bl, cl, dl, el, w14, 5);
    RL_J(al, bl, cl, dl, el, w1, 12);
    RL_J(al, bl, cl, dl, el, w3, 13);
    RL_J(al, bl, cl, dl, el, w8, 14);
    RL_J(al, bl, cl, dl, el, w11, 11);
    RL_J(al, bl, cl, dl, el, w6, 8);
    RL_J(al, bl, cl, dl, el, w15, 5);
    RL_J(al, bl, cl, dl, el, w13, 6);

    /* Right chain: rounds 0-15, J(b,c,d) = b^(c|~d) */
    RR_J(ar, br, cr, dr, er, w5, 8);
    RR_J(ar, br, cr, dr, er, w14, 9);
    RR_J(ar, br, cr, dr, er, w7, 9);
    RR_J(ar, br, cr, dr, er, w0, 11);
    RR_J(ar, br, cr, dr, er, w9, 13);
    RR_J(ar, br, cr, dr, er, w2, 15);
    RR_J(ar, br, cr, dr, er, w11, 15);
    RR_J(ar, br, cr, dr, er, w4, 5);
    RR_J(ar, br, cr, dr, er, w13, 7);
    RR_J(ar, br, cr, dr, er, w6, 7);
    RR_J(ar, br, cr, dr, er, w15, 8);
    RR_J(ar, br, cr, dr, er, w8, 11);
    RR_J(ar, br, cr, dr, er, w1, 14);
    RR_J(ar, br, cr, dr, er, w10, 14);
    RR_J(ar, br, cr, dr, er, w3, 12);
    RR_J(ar, br, cr, dr, er, w12, 6);
    /* Right chain: rounds 16-31, I(b,c,d) = (b&d)|(c&~d) */
    RR_I(ar, br, cr, dr, er, w6, 9);
    RR_I(ar, br, cr, dr, er, w11, 13);
    RR_I(ar, br, cr, dr, er, w3, 15);
    RR_I(ar, br, cr, dr, er, w7, 7);
    RR_I(ar, br, cr, dr, er, w0, 12);
    RR_I(ar, br, cr, dr, er, w13, 8);
    RR_I(ar, br, cr, dr, er, w5, 9);
    RR_I(ar, br, cr, dr, er, w10, 11);
    RR_I(ar, br, cr, dr, er, w14, 7);
    RR_I(ar, br, cr, dr, er, w15, 7);
    RR_I(ar, br, cr, dr, er, w8, 12);
    RR_I(ar, br, cr, dr, er, w12, 7);
    RR_I(ar, br, cr, dr, er, w4, 6);
    RR_I(ar, br, cr, dr, er, w9, 15);
    RR_I(ar, br, cr, dr, er, w1, 13);
    RR_I(ar, br, cr, dr, er, w2, 11);
    /* Right chain: rounds 32-47, H(b,c,d) = (b|~c)^d */
    RR_H(ar, br, cr, dr, er, w15, 9);
    RR_H(ar, br, cr, dr, er, w5, 7);
    RR_H(ar, br, cr, dr, er, w1, 15);
    RR_H(ar, br, cr, dr, er, w3, 11);
    RR_H(ar, br, cr, dr, er, w7, 8);
    RR_H(ar, br, cr, dr, er, w14, 6);
    RR_H(ar, br, cr, dr, er, w6, 6);
    RR_H(ar, br, cr, dr, er, w9, 14);
    RR_H(ar, br, cr, dr, er, w11, 12);
    RR_H(ar, br, cr, dr, er, w8, 13);
    RR_H(ar, br, cr, dr, er, w12, 5);
    RR_H(ar, br, cr, dr, er, w2, 14);
    RR_H(ar, br, cr, dr, er, w10, 13);
    RR_H(ar, br, cr, dr, er, w0, 13);
    RR_H(ar, br, cr, dr, er, w4, 7);
    RR_H(ar, br, cr, dr, er, w13, 5);
    /* Right chain: rounds 48-63, G(b,c,d) = (b&c)|(~b&d) */
    RR_G(ar, br, cr, dr, er, w8, 15);
    RR_G(ar, br, cr, dr, er, w6, 5);
    RR_G(ar, br, cr, dr, er, w4, 8);
    RR_G(ar, br, cr, dr, er, w1, 11);
    RR_G(ar, br, cr, dr, er, w3, 14);
    RR_G(ar, br, cr, dr, er, w11, 14);
    RR_G(ar, br, cr, dr, er, w15, 6);
    RR_G(ar, br, cr, dr, er, w0, 14);
    RR_G(ar, br, cr, dr, er, w5, 6);
    RR_G(ar, br, cr, dr, er, w12, 9);
    RR_G(ar, br, cr, dr, er, w2, 12);
    RR_G(ar, br, cr, dr, er, w13, 9);
    RR_G(ar, br, cr, dr, er, w9, 12);
    RR_G(ar, br, cr, dr, er, w7, 5);
    RR_G(ar, br, cr, dr, er, w10, 15);
    RR_G(ar, br, cr, dr, er, w14, 8);
    /* Right chain: rounds 64-79, F(b,c,d) = b^c^d */
    RR_F(ar, br, cr, dr, er, w12, 8);
    RR_F(ar, br, cr, dr, er, w15, 5);
    RR_F(ar, br, cr, dr, er, w10, 12);
    RR_F(ar, br, cr, dr, er, w4, 9);
    RR_F(ar, br, cr, dr, er, w1, 12);
    RR_F(ar, br, cr, dr, er, w5, 5);
    RR_F(ar, br, cr, dr, er, w8, 14);
    RR_F(ar, br, cr, dr, er, w7, 6);
    RR_F(ar, br, cr, dr, er, w6, 8);
    RR_F(ar, br, cr, dr, er, w2, 13);
    RR_F(ar, br, cr, dr, er, w13, 6);
    RR_F(ar, br, cr, dr, er, w14, 5);
    RR_F(ar, br, cr, dr, er, w0, 15);
    RR_F(ar, br, cr, dr, er, w3, 13);
    RR_F(ar, br, cr, dr, er, w9, 11);
    RR_F(ar, br, cr, dr, er, w11, 11);

    uint32_t t = state[1] + cl + dr;
    state[1] = state[2] + dl + er;
    state[2] = state[3] + el + ar;
    state[3] = state[4] + al + br;
    state[4] = state[0] + bl + cr;
    state[0] = t;
}

void ripemd160_init(ripemd160_ctx *ctx)
{
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    ctx->count[0] = ctx->count[1] = 0;
}

void ripemd160_update(ripemd160_ctx *ctx, const uint8_t *data, size_t len)
{
    size_t have = (ctx->count[0] >> 3) & 63;
    if ((ctx->count[0] += (uint32_t)(len << 3)) < (uint32_t)(len << 3))
        ctx->count[1]++;
    ctx->count[1] += (uint32_t)(len >> 29);
    size_t need = 64 - have;
    size_t off = 0;
    if (len >= need) {
        memcpy(ctx->buf + have, data, need);
        ripemd160_compress(ctx->state, ctx->buf);
        off = need; have = 0;
        while (off + 64 <= len) {
            ripemd160_compress(ctx->state, data + off);
            off += 64;
        }
    }
    memcpy(ctx->buf + have, data + off, len - off);
}

void ripemd160_final(ripemd160_ctx *ctx, uint8_t *digest)
{
    uint8_t pad[64] = {0x80};
    uint8_t len_buf[8];
    for (int i = 0; i < 4; i++) {
        len_buf[i]   = (ctx->count[0] >> (i*8)) & 0xFF;
        len_buf[i+4] = (ctx->count[1] >> (i*8)) & 0xFF;
    }
    size_t have = (ctx->count[0] >> 3) & 63;
    size_t pad_len = (have < 56) ? (56 - have) : (120 - have);
    ripemd160_update(ctx, pad, pad_len);
    ripemd160_update(ctx, len_buf, 8);
    for (int i = 0; i < 5; i++) {
        digest[i*4]   = (ctx->state[i])       & 0xFF;
        digest[i*4+1] = (ctx->state[i] >>  8) & 0xFF;
        digest[i*4+2] = (ctx->state[i] >> 16) & 0xFF;
        digest[i*4+3] = (ctx->state[i] >> 24) & 0xFF;
    }
}

void ripemd160(const uint8_t *data, size_t len, uint8_t *digest)
{
    ripemd160_ctx ctx;
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, data, len);
    ripemd160_final(&ctx, digest);
}

/*
 * Specialized RIPEMD160 for fixed 32-byte input, eliminates ctx/update/final overhead
 */
void ripemd160_32(const uint8_t *data32, uint8_t *digest)
{
    /* Initial state */
    uint32_t state[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };

    /* Construct padded block: 32-byte data + 0x80 + zero padding + length (little-endian, 256 bits) */
    uint8_t block[64];
    memcpy(block, data32, 32);
    block[32] = 0x80;
    memset(block + 33, 0, 64 - 33 - 8);
    /* Length field: 256 bits = 0x100, written little-endian into last 8 bytes */
    block[56] = 0x00; block[57] = 0x01;
    block[58] = 0x00; block[59] = 0x00;
    block[60] = 0x00; block[61] = 0x00;
    block[62] = 0x00; block[63] = 0x00;

    ripemd160_compress(state, block);

    /* Output (little-endian) */
    for (int i = 0; i < 5; i++) {
        digest[i * 4] = (state[i]) & 0xFF;
        digest[i * 4 + 1] = (state[i] >> 8) & 0xFF;
        digest[i * 4 + 2] = (state[i] >> 16) & 0xFF;
        digest[i * 4 + 3] = (state[i] >> 24) & 0xFF;
    }
}

