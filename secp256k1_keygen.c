/*
 * secp256k1_keygen.c
 *
 * 封装libsecp256k1内部接口，实现：
 *   1. 直接点加法（绕过ecmult路径）
 *   2. 批量仿射坐标归一化（Montgomery trick）
 *   3. 直接从仿射坐标构造公钥字节（跳过 serialize）
 *
 * 编译模式：
 *   - 默认：使用内部头文件，链接本地编译的 secp256k1_lib.o
 *   - USE_PUBKEY_API_ONLY：回退到公开API
 *
 * 注意：本文件定义 SECP256K1_BUILD 并包含 *_impl.h，
 * 这些 static/inline 函数每个编译单元都需要自己的副本，
 * 不会与 secp256k1_lib.o 产生符号冲突（static 符号不导出）。
 * secp256k1_context_struct 等非 static 符号由 secp256k1_lib.o 提供。
 */

/*
 * 注意：SECP256K1_BUILD 和所有 *_impl.h 已在 secp256k1_keygen.h 中定义/包含。
 * 此处直接包含 secp256k1_keygen.h 即可，无需重复定义。
 */
#include "secp256k1_keygen.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef USE_PUBKEY_API_ONLY

/*
 * secp256k1_context_struct 完整定义（与 secp256k1/src/secp256k1.c 中一致）。
 * 此处需要完整定义才能访问 ctx->ecmult_gen_ctx 字段。
 * 结构体定义不产生链接符号，与 secp256k1_lib.o 中的定义不冲突。
 */
struct secp256k1_context_struct {
    secp256k1_ecmult_gen_context ecmult_gen_ctx;
    secp256k1_callback illegal_callback;
    secp256k1_callback error_callback;
    int declassify;
};

int keygen_init_generator(const secp256k1_context *ctx,
                          secp256k1_ge *G_out)
{
    (void)ctx;

    /*
     * secp256k1生成元G的仿射坐标（硬编码标准值）
     * X = 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
     * Y = 483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
     */
    static const unsigned char Gx[32] = {
        0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
        0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
        0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
        0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
    };
    static const unsigned char Gy[32] = {
        0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,
        0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
        0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,
        0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8
    };

    if (!secp256k1_fe_set_b32_limit(&G_out->x, Gx)) return -1;
    if (!secp256k1_fe_set_b32_limit(&G_out->y, Gy)) return -1;
    G_out->infinity = 0;

    /* 验证 infinity 标志 */
    if (G_out->infinity != 0) {
        fprintf(stderr, "keygen_init_generator: G.infinity != 0，异常！\n");
        return -1;
    }
    return 0;
}

int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_gej *gej_out)
{
    secp256k1_scalar scalar;
    int overflow = 0;

    secp256k1_scalar_set_b32(&scalar, privkey, &overflow);
    if (overflow || secp256k1_scalar_is_zero(&scalar)) {
        return -1;
    }

    /* 直接调用内部ecmult_gen：gej = scalar * G */
    secp256k1_ecmult_gen(&ctx->ecmult_gen_ctx, gej_out, &scalar);

    /* 清零标量，防止侧信道泄露 */
    secp256k1_scalar_clear(&scalar);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 使用Montgomery trick批量归一化：                                     */
/*   acc[0] = Z[0]                                                     */
/*   acc[i] = acc[i-1] * Z[i]                                         */
/*   inv = 1 / acc[n-1]  （1次模逆）                                   */
/*   从后往前：                                                         */
/*     inv_zi = inv * acc[i-1]  （Z[i]的逆）                           */
/*     inv    = inv * Z[i]      （更新inv为acc[i-1]的逆）               */
/* ------------------------------------------------------------------ */

/* 大小与keysearch.c中的BATCH_SIZE保持一致 */
#define KEYGEN_BATCH_MAX (4096)
static __thread secp256k1_fe acc_buf[KEYGEN_BATCH_MAX];

void keygen_batch_normalize(const secp256k1_gej *gej_in,
                            secp256k1_ge *ge_out,
                            size_t n)
{
    if (n == 0 || n > KEYGEN_BATCH_MAX)
        return;

    /* 优先使用线程局部静态缓冲区，避免堆分配 */
    secp256k1_fe *acc = acc_buf;

    /* 第一步：计算累积乘积，跳过infinity点 */
    int first = 1;
    size_t first_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (gej_in[i].infinity) {
            ge_out[i].infinity = 1;
            /* acc[i]留作占位，不参与运算 */
            if (!first) {
                acc[i] = acc[i - 1];
            } else {
                secp256k1_fe_set_int(&acc[i], 1);
            }
            continue;
        }
        if (first) {
            acc[i] = gej_in[i].z;
            first = 0;
            first_idx = i;
        } else {
            secp256k1_fe_mul(&acc[i], &acc[i - 1], &gej_in[i].z);
        }
    }

    if (first) {
        /* 所有点都是infinity */
        return;
    }

    /* 第二步：计算acc[n-1]的模逆 */
    secp256k1_fe inv;
    secp256k1_fe_inv(&inv, &acc[n - 1]);

    /* 第三步：从后往前，逐个恢复Z的逆，并转换为仿射坐标 */
    for (size_t i = n; i-- > 0; ) {
        if (gej_in[i].infinity) {
            continue;
        }

        secp256k1_fe inv_zi;
        if (i == first_idx) {
            /* 第一个非infinity点，inv已经是Z[i]的逆 */
            inv_zi = inv;
        } else {
            /* inv_zi = inv * acc[i-1] */
            secp256k1_fe_mul(&inv_zi, &inv, &acc[i - 1]);
            /* 更新inv = inv * Z[i]，使其成为acc[i-1]的逆 */
            secp256k1_fe_mul(&inv, &inv, &gej_in[i].z);
        }

        /* 仿射坐标：x = X/Z^2, y = Y/Z^3 */
        secp256k1_fe inv_zi2;
        secp256k1_fe_sqr(&inv_zi2, &inv_zi);

        secp256k1_fe_mul(&ge_out[i].x, &gej_in[i].x, &inv_zi2);
        secp256k1_fe_mul(&inv_zi2, &inv_zi2, &inv_zi);   /* inv_zi^3 */
        secp256k1_fe_mul(&ge_out[i].y, &gej_in[i].y, &inv_zi2);
        ge_out[i].infinity = 0;
    }
}

/*
 * keygen_batch_normalize_rzr：利用rzr增量因子加速的批量归一化
 * 原理：
 *   gej_in[i]由gej_in[i-1] + G 得到，secp256k1_gej_add_ge_var输出的rzr[i-1]
 *   满足Z[i] = Z[i-1] * rzr[i-1]，因此：
 *     1/Z[i-1] = (1/Z[i]) * rzr[i-1]
 *
 *   直接对gej_in[n-1].z求模逆（1次），然后从后往前用rzr链推导每个点的1/Z[i]：
 *     inv = 1/Z[n-1]
 *     inv_zi = inv  （第n-1个点）
 *     inv = inv * rzr[i-1]  →  inv = 1/Z[i-1]  （向前传递）
 *
 * 要求：所有点均非infinity（内层循环已保证）
 * 参数：
 *   gej_in  : 输入Jacobian坐标数组（大小n）
 *   ge_out  : 输出仿射坐标数组（大小>=n）
 *   rzr     : Z坐标增量因子数组（大小n-1，rzr[i]满足Z[i+1]=Z[i]*rzr[i]）
 *   n       : 数组元素个数
 */
void keygen_batch_normalize_rzr(const secp256k1_gej *gej_in,
                                secp256k1_ge *ge_out,
                                const secp256k1_fe *rzr,
                                size_t n)
{
    if (n == 0 || n > KEYGEN_BATCH_MAX)
        return;

    /*
     * 直接对最后一个点的Z坐标求模逆（1次模逆）
     * 无需前向累积：gej_in[n-1].z就是所有rzr乘积的终点
     */
    secp256k1_fe inv;
    secp256k1_fe_inv(&inv, &gej_in[n - 1].z);

    /*
     * 从后往前，利用rzr链逐个推导1/Z[i]，并转换为仿射坐标：
     *   inv始终持有当前点i的1/Z[i]
     *   处理完第i点后：inv = inv * rzr[i-1] = 1/Z[i-1]
     */
    for (size_t i = n; i-- > 0; ) {
        /* inv 此时 = 1/Z[i] */
        secp256k1_fe inv_zi2;
        secp256k1_fe_sqr(&inv_zi2, &inv);                           /* inv_zi^2 */
        secp256k1_fe_mul(&ge_out[i].x, &gej_in[i].x, &inv_zi2);     /* X/Z^2 */
        secp256k1_fe_mul(&inv_zi2, &inv_zi2, &inv);                 /* inv_zi^3 */
        secp256k1_fe_mul(&ge_out[i].y, &gej_in[i].y, &inv_zi2);     /* Y/Z^3 */
        ge_out[i].infinity = 0;

        /* 向前传递：inv = 1/Z[i-1] = (1/Z[i]) * rzr[i-1] */
        if (i > 0) {
            secp256k1_fe_mul(&inv, &inv, &rzr[i - 1]);
        }
    }
}

void keygen_ge_to_pubkey_bytes(const secp256k1_ge *ge,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out)
{
    if (ge->infinity)
        return;

    secp256k1_fe x = ge->x;
    secp256k1_fe y = ge->y;

    uint8_t x_bytes[32];
    uint8_t y_bytes[32];
    secp256k1_fe_get_b32(x_bytes, &x);

    if (compressed_out != NULL) {
        /* 压缩公钥：前缀0x02（Y偶）或0x03（Y奇）+ X(32字节) */
        compressed_out[0] = secp256k1_fe_is_odd(&y) ? 0x03 : 0x02;
        memcpy(compressed_out + 1, x_bytes, 32);
    }

    if (uncompressed_out != NULL) {
        secp256k1_fe_get_b32(y_bytes, &y);
        /* 非压缩公钥：0x04 + X(32字节) + Y(32字节) */
        uncompressed_out[0] = 0x04;
        memcpy(uncompressed_out + 1, x_bytes, 32);
        memcpy(uncompressed_out + 1 + 32, y_bytes, 32);
    }
}

#else

int keygen_init_generator(const secp256k1_context *ctx,
                          void *G_out)
{
    (void)ctx;
    (void)G_out;
    /* 回退模式无需预加载G，直接返回成功 */
    return 0;
}

int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_pubkey *pubkey_out)
{
    if (!secp256k1_ec_pubkey_create(ctx, pubkey_out, privkey))
        return -1;
    return 0;
}

void keygen_ge_to_pubkey_bytes(const secp256k1_context *ctx,
                               const secp256k1_pubkey *pubkey,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out)
{
    if (compressed_out != NULL) {
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(ctx, compressed_out, &len,
                                      pubkey, SECP256K1_EC_COMPRESSED);
    }
    if (uncompressed_out != NULL) {
        size_t len = 65;
        secp256k1_ec_pubkey_serialize(ctx, uncompressed_out, &len,
                                      pubkey, SECP256K1_EC_UNCOMPRESSED);
    }
}

#endif /* USE_PUBKEY_API_ONLY */

