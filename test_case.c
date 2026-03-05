#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sha256.h"
#include "ripemd160.h"
#include "hash_utils.h"
/* secp256k1_keygen.h 在内部模式下已包含 secp256k1.h（源码目录版本），
 * 回退模式下需要系统 secp256k1.h，通过条件编译处理 */
#ifndef USE_PUBKEY_API_ONLY
#  include "secp256k1_keygen.h"
#else
#  include <secp256k1.h>
#  include "secp256k1_keygen.h"
#endif

/* ===================== 辅助函数 ===================== */

static int pass_count = 0;
static int fail_count = 0;

/* secp256k1上下文（只读，多线程安全） */
secp256k1_context *secp_ctx = NULL;

#ifndef USE_PUBKEY_API_ONLY
/* 全局生成元 G 的仿射坐标（由 keygen_init_generator 初始化） */
secp256k1_ge G_affine;
#endif

/* 将字节数组转为小写十六进制字符串（out 需至少 len*2+1 字节） */
static void bytes_to_hex_helper(const uint8_t *buf, size_t len, char *out) {
    for (size_t i = 0; i < len; i++) {
        sprintf(out + i * 2, "%02x", buf[i]);
    }
    out[len * 2] = '\0';
}

/* 断言函数：比较期望十六进制字符串与实际字节数组 */
static void check(const char *name, const char *expected_hex,
                  const uint8_t *actual, size_t len) {
    char actual_hex[len * 2 + 1];
    bytes_to_hex_helper(actual, len, actual_hex);
    if (strcmp(expected_hex, actual_hex) == 0) {
        printf("  [PASS] %s\n", name);
        pass_count++;
    } else {
        printf("  [FAIL] %s\n", name);
        printf("         期望: %s\n", expected_hex);
        printf("         实际: %s\n", actual_hex);
        fail_count++;
    }
}

/* ===================== SHA256 测试 ===================== */

static void test_sha256(void) {
    printf("\n=== SHA256 标准测试向量 ===\n");

    sha256_ctx ctx;
    uint8_t digest[32];
    uint8_t buf[128];

    /* 1.1 空串 */
    sha256_init(&ctx);
    sha256_final(&ctx, digest);
    check("SHA256(\"\") 空串",
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          digest, 32);

    /* 1.2 "abc" */
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)"abc", 3);
    sha256_final(&ctx, digest);
    check("SHA256(\"abc\")",
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          digest, 32);

    /* 1.3 448bit 跨块消息 */
    const char *msg448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)msg448, strlen(msg448));
    sha256_final(&ctx, digest);
    check("SHA256(448bit 跨块消息)",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          digest, 32);

    /* 1.4 55字节全 'a' */
    memset(buf, 'a', 55);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 55);
    sha256_final(&ctx, digest);
    check("SHA256('a'*55)",
          "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
          digest, 32);

    /* 1.5 56字节全 'a' */
    memset(buf, 'a', 56);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 56);
    sha256_final(&ctx, digest);
    check("SHA256('a'*56)",
          "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
          digest, 32);

    /* 1.6 64字节全 'a' */
    memset(buf, 'a', 64);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 64);
    sha256_final(&ctx, digest);
    check("SHA256('a'*64)",
          "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
          digest, 32);

    /* 1.7 65字节全 'a' */
    memset(buf, 'a', 65);
    sha256_init(&ctx);
    sha256_update(&ctx, buf, 65);
    sha256_final(&ctx, digest);
    check("SHA256('a'*65)",
          "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
          digest, 32);
}

/* ===================== RIPEMD160 测试 ===================== */

static void test_ripemd160(void) {
    printf("\n=== RIPEMD160 标准测试向量 ===\n");

    ripemd160_ctx ctx;
    uint8_t digest[20];
    uint8_t buf[128];

    /* 2.1 空串 */
    ripemd160_init(&ctx);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"\") 空串",
          "9c1185a5c5e9fc54612808977ee8f548b2258d31",
          digest, 20);

    /* 2.2 "abc" */
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)"abc", 3);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"abc\")",
          "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
          digest, 20);

    /* 2.3 26字节字母表 */
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26);
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(\"abcdefghijklmnopqrstuvwxyz\")",
          "f71c27109c692c1b56bbdceb5b9d2865b3708dbc",
          digest, 20);

    /* 2.4 56字节跨块消息 */
    const char *msg56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, (const uint8_t *)msg56, strlen(msg56));
    ripemd160_final(&ctx, digest);
    check("RIPEMD160(56字节跨块消息)",
          "12a053384a9c0c88e405a06c27dcf49ada62eb2b",
          digest, 20);

    /* 2.5 55字节全 'a' */
    memset(buf, 'a', 55);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 55);
    ripemd160_final(&ctx, digest);
    /* 自一致性：先用 ripemd160() 便捷函数计算参考值 */
    uint8_t ref[20];
    ripemd160(buf, 55, ref);
    char ref_hex[41];
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160('a'*55) 自一致性", ref_hex, digest, 20);

    /* 2.6 64字节全 'a' */
    memset(buf, 'a', 64);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 64);
    ripemd160_final(&ctx, digest);
    ripemd160(buf, 64, ref);
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160('a'*64) 自一致性", ref_hex, digest, 20);

    /* 2.7 32字节全零 */
    memset(buf, 0x00, 32);
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, buf, 32);
    ripemd160_final(&ctx, digest);
    ripemd160(buf, 32, ref);
    bytes_to_hex_helper(ref, 20, ref_hex);
    check("RIPEMD160(0x00*32) 自一致性", ref_hex, digest, 20);
}

/* ===================== Hash160 组合测试 ===================== */

static void test_hash160(void) {
    printf("\n=== Hash160 组合场景验证 ===\n");

    uint8_t sha_digest[32];
    uint8_t rmd_digest[20];

    /*
     * 3.1 已知 33字节压缩公钥（比特币创世区块 coinbase 公钥）
     *   公钥: 04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb6
     *         49f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f
     *   压缩公钥 (33字节): 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   对应 Hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
     */
    static const uint8_t compressed_pubkey[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17,
        0x98
    };
    sha256(compressed_pubkey, 33, sha_digest);
    ripemd160(sha_digest, 32, rmd_digest);
    check("Hash160(压缩公钥 33字节)",
          "751e76e8199196d454941c45d1b3a323f1433bd6",
          rmd_digest, 20);

    /*
     * 3.2 已知 65字节非压缩公钥（同一密钥的非压缩形式）
     *   非压缩公钥 (65字节): 04 + X(32) + Y(32)
     *   对应 Hash160: 91b24bf9f5288532960ac687abb035127b1d28a5
     */
    static const uint8_t uncompressed_pubkey[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };
    sha256(uncompressed_pubkey, 65, sha_digest);
    ripemd160(sha_digest, 32, rmd_digest);
    check("Hash160(非压缩公钥 65字节)",
          "91b24bf9f5288532960ac687abb035127b1d28a5",
          rmd_digest, 20);
}

/* ===================== 增量公钥推导测试 ===================== */

/*
 * 辅助：用私钥直接计算压缩公钥序列化字节
 */
static void privkey_to_compressed_bytes(const uint8_t *privkey, uint8_t out[33])
{
    secp256k1_pubkey pubkey;
    size_t len = 33;
    secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
    secp256k1_ec_pubkey_serialize(secp_ctx, out, &len, &pubkey,
                                  SECP256K1_EC_COMPRESSED);
}

static void test_incremental_pubkey(void) {
    printf("\n=== 增量公钥推导正确性测试 ===\n");

    /* tweak = 1（每步私钥 +1，公钥点加 G） */
    uint8_t tweak[32] = {0};
    tweak[31] = 1;

    /* ------------------------------------------------------------------ */
    /* 4.1  单步增量：k=1 -> k'=2
     *   私钥 k  = 0x00...01  对应压缩公钥 P
     *   私钥 k' = 0x00...02  对应压缩公钥 P'（直接计算）
     *   增量推导：P_incr = P + G（通过 secp256k1_ec_pubkey_tweak_add）
     *   期望：P' == P_incr
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey[32] = {0};
        privkey[31] = 1;  /* k = 1 */

        /* 直接计算 k'=2 的公钥 */
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;
        uint8_t direct_bytes[33];
        privkey_to_compressed_bytes(privkey2, direct_bytes);

        /* 增量推导：从 k=1 的公钥点加 G */
        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
        secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);
        uint8_t incr_bytes[33];
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, incr_bytes, &len, &pubkey,
                                      SECP256K1_EC_COMPRESSED);

        char direct_hex[67];
        bytes_to_hex_helper(direct_bytes, 33, direct_hex);
        check("4.1 单步增量公钥（k=1 -> k'=2）与直接计算一致",
              direct_hex, incr_bytes, 33);
    }

    /* ------------------------------------------------------------------ */
    /* 4.2  单步增量后的 hash160 一致性
     *   验证：pubkey_bytes_to_hash160(P_incr) == privkey_to_hash160(k')
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;  /* k' = 2 */

        /* 方法A：privkey_to_hash160 直接从私钥计算 */
        uint8_t hash160_direct_comp[20];
        uint8_t hash160_direct_uncomp[20];
        privkey_to_hash160(privkey2, hash160_direct_comp, hash160_direct_uncomp);

        /* 方法B：增量推导公钥后用 pubkey_bytes_to_hash160 */
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;
        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey1);
        secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);

        uint8_t comp_bytes[33];
        uint8_t uncomp_bytes[65];
        size_t len_c = 33, len_u = 65;
        secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes,   &len_c, &pubkey,
                                      SECP256K1_EC_COMPRESSED);
        secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bytes, &len_u, &pubkey,
                                      SECP256K1_EC_UNCOMPRESSED);

        uint8_t hash160_incr_comp[20];
        uint8_t hash160_incr_uncomp[20];
        pubkey_bytes_to_hash160(comp_bytes,   33, hash160_incr_comp);
        pubkey_bytes_to_hash160(uncomp_bytes, 65, hash160_incr_uncomp);

        /* 转为 hex 做比对 */
        char direct_comp_hex[41];
        bytes_to_hex_helper(hash160_direct_comp, 20, direct_comp_hex);
        check("4.2a 增量推导压缩公钥 hash160 与 privkey_to_hash160 一致（k'=2）",
              direct_comp_hex, hash160_incr_comp, 20);

        char direct_uncomp_hex[41];
        bytes_to_hex_helper(hash160_direct_uncomp, 20, direct_uncomp_hex);
        check("4.2b 增量推导非压缩公钥 hash160 与 privkey_to_hash160 一致（k'=2）",
              direct_uncomp_hex, hash160_incr_uncomp, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 4.3  多步增量（10步）：从 k=1 连续推导到 k=11
     *   每步验证增量推导的压缩公钥 hash160 与直接计算一致
     * ------------------------------------------------------------------ */
    {
        printf("  [多步增量推导 10步，k=1..11]\n");
        uint8_t privkey[32] = {0};
        privkey[31] = 1;

        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);

        int all_pass = 1;
        for (int step = 1; step <= 10; step++) {
            /* 增量推导：私钥 +1，公钥点加 G */
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak);
            secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);

            /* 增量推导的 hash160 */
            uint8_t comp_bytes[33];
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            uint8_t hash160_incr[20];
            pubkey_bytes_to_hash160(comp_bytes, 33, hash160_incr);

            /* 直接计算的 hash160 */
            uint8_t hash160_direct[20];
            privkey_to_hash160(privkey, hash160_direct, NULL);

            if (memcmp(hash160_incr, hash160_direct, 20) != 0) {
                char incr_hex[41], direct_hex[41];
                bytes_to_hex_helper(hash160_incr,   20, incr_hex);
                bytes_to_hex_helper(hash160_direct, 20, direct_hex);
                printf("  [FAIL] 步骤 %d: 增量=%s 直接=%s\n",
                       step, incr_hex, direct_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 4.3 多步增量推导 hash160（10步全部一致）\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 4.4  已知向量：私钥 k=1 的压缩公钥 hash160
     *   私钥 k=1 的压缩公钥:
     *     0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   对应 hash160: 751e76e8199196d454941c45d1b3a323f1433bd6
     *   （与 test_hash160 中 3.1 使用的公钥相同，交叉验证）
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        uint8_t hash160_comp[20];
        privkey_to_hash160(privkey1, hash160_comp, NULL);
        check("4.4 已知向量：k=1 压缩公钥 hash160",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160_comp, 20);

        /* 同时用 pubkey_bytes_to_hash160 验证 */
        uint8_t comp_bytes[33];
        privkey_to_compressed_bytes(privkey1, comp_bytes);
        uint8_t hash160_via_bytes[20];
        pubkey_bytes_to_hash160(comp_bytes, 33, hash160_via_bytes);
        check("4.4b pubkey_bytes_to_hash160(k=1 压缩公钥) 与已知向量一致",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160_via_bytes, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 4.5  BATCH_SIZE 边界：连续推导 2048 步后重置，验证重置后首步正确性
     *   基准私钥 k=0x00...05，推导 2048 步后得到 k'=k+2048
     *   验证增量推导的 hash160 与直接计算 k' 的 hash160 一致
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey[32] = {0};
        privkey[31] = 5;  /* k = 5 */

        secp256k1_pubkey pubkey;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);

        /* 连续推导 2048 步 */
        for (int i = 0; i < 2048; i++) {
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak);
            secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak);
        }

        /* 增量推导结果 */
        uint8_t comp_bytes[33];
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(secp_ctx, comp_bytes, &len, &pubkey,
                                      SECP256K1_EC_COMPRESSED);
        uint8_t hash160_incr[20];
        pubkey_bytes_to_hash160(comp_bytes, 33, hash160_incr);

        /* 直接计算 k+2048 的 hash160 */
        uint8_t hash160_direct[20];
        privkey_to_hash160(privkey, hash160_direct, NULL);

        char direct_hex[41];
        bytes_to_hex_helper(hash160_direct, 20, direct_hex);
        check("4.5 BATCH_SIZE(2048步)边界：增量推导 hash160 与直接计算一致",
              direct_hex, hash160_incr, 20);
    }
}

/* ===================== AVX2 压缩函数测试 ===================== */

#ifdef __AVX2__

/*
 * 辅助：构造 SHA256 padded block（单块，消息 < 56 字节）
 *   SHA256 padding：消息 + 0x80 + 零填充 + 8字节大端消息位长
 *   block 为大端序 uint32_t，直接作为 sha256_compress 的输入
 */
static void make_sha256_padded_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    /* 消息位长（大端序）写入最后 8 字节 */
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen >> 56);
    block[57] = (uint8_t)(bitlen >> 48);
    block[58] = (uint8_t)(bitlen >> 40);
    block[59] = (uint8_t)(bitlen >> 32);
    block[60] = (uint8_t)(bitlen >> 24);
    block[61] = (uint8_t)(bitlen >> 16);
    block[62] = (uint8_t)(bitlen >>  8);
    block[63] = (uint8_t)(bitlen      );
}

/*
 * 辅助：构造 RIPEMD160 padded block（单块，消息 < 56 字节）
 *   RIPEMD160 padding：消息 + 0x80 + 零填充 + 8字节小端消息位长
 */
static void make_ripemd160_padded_block(const uint8_t *msg, size_t msglen, uint8_t block[64])
{
    memset(block, 0, 64);
    memcpy(block, msg, msglen);
    block[msglen] = 0x80;
    /* 消息位长（小端序）写入最后 8 字节 */
    uint64_t bitlen = (uint64_t)msglen * 8;
    block[56] = (uint8_t)(bitlen      );
    block[57] = (uint8_t)(bitlen >>  8);
    block[58] = (uint8_t)(bitlen >> 16);
    block[59] = (uint8_t)(bitlen >> 24);
    block[60] = (uint8_t)(bitlen >> 32);
    block[61] = (uint8_t)(bitlen >> 40);
    block[62] = (uint8_t)(bitlen >> 48);
    block[63] = (uint8_t)(bitlen >> 56);
}

/* SHA256 初始状态常量 */
static const uint32_t SHA256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* RIPEMD160 初始状态常量 */
static const uint32_t RMD160_INIT[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

/*
 * 辅助：用标量 sha256_compress（通过 sha256_ctx 内部机制）计算单块压缩结果
 * 方法：用 sha256_init + sha256_update + sha256_final 的完整流程，
 *       对于单块消息（< 56 字节），final 后的 state 即为压缩结果。
 * 这里直接用 sha256() 便捷函数计算完整哈希，作为 AVX2 结果的参考。
 */

/*
 * 辅助：将 state[8] 转为 32 字节大端序摘要（SHA256 输出格式）
 */
static void sha256_state_to_digest(const uint32_t state[8], uint8_t digest[32])
{
    for (int i = 0; i < 8; i++) {
        digest[i*4+0] = (uint8_t)(state[i] >> 24);
        digest[i*4+1] = (uint8_t)(state[i] >> 16);
        digest[i*4+2] = (uint8_t)(state[i] >>  8);
        digest[i*4+3] = (uint8_t)(state[i]      );
    }
}

/*
 * 辅助：将 state[5] 转为 20 字节小端序摘要（RIPEMD160 输出格式）
 */
static void rmd160_state_to_digest(const uint32_t state[5], uint8_t digest[20])
{
    for (int i = 0; i < 5; i++) {
        digest[i*4+0] = (uint8_t)(state[i]      );
        digest[i*4+1] = (uint8_t)(state[i] >>  8);
        digest[i*4+2] = (uint8_t)(state[i] >> 16);
        digest[i*4+3] = (uint8_t)(state[i] >> 24);
    }
}

static void test_avx2_compress(void) {
    printf("\n=== AVX2 压缩函数测试（sha256_compress_avx2 / ripemd160_compress_avx2）===\n");

    /* ------------------------------------------------------------------ */
    /* 7.1  sha256_compress_avx2 — 8路相同消息，结果与标量 sha256() 一致
     *
     * 测试方法：
     *   - 构造 8 个相同的 padded block（对应 "abc"）
     *   - 用 AVX2 函数压缩，得到 8 路 state
     *   - 将 state 转为摘要，与 sha256("abc") 标准值对比
     * ------------------------------------------------------------------ */
    {
        /* "abc" 的 padded block */
        uint8_t block_abc[64];
        make_sha256_padded_block((const uint8_t *)"abc", 3, block_abc);

        /* 8 路 state，全部初始化为 SHA256 初始值 */
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];
        for (int i = 0; i < 8; i++) {
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        sha256_compress_avx2(states, blocks);

        /* 将每路 state 转为摘要，与标准值对比 */
        uint8_t digest_avx2[32];
        for (int lane = 0; lane < 8; lane++) {
            sha256_state_to_digest(states_data[lane], digest_avx2);
            char name[64];
            snprintf(name, sizeof(name), "7.1 sha256_compress_avx2(\"abc\") lane%d 与标准值一致", lane);
            check(name,
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                  digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.2  sha256_compress_avx2 — 8路不同消息，每路结果与标量 sha256() 一致
     *
     * 8 路消息：空串、"abc"、全零1字节、全0xFF 1字节、
     *           "hello"、"world"、递增序列(8字节)、全'a'*10
     * ------------------------------------------------------------------ */
    {
        /* 准备 8 个不同的 padded block */
        static const struct { const uint8_t *msg; size_t len; const char *expected; } cases[8] = {
            { (const uint8_t *)"",          0,  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
            { (const uint8_t *)"abc",       3,  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
            { (const uint8_t *)"\x00",      1,  "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d" },
            { (const uint8_t *)"\xff",      1,  "a8100ae6aa1940d0b663bb31cd466142ebbdbd5187131b92d93818987832eb89" },
            { (const uint8_t *)"hello",     5,  "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824" },
            { (const uint8_t *)"world",     5,  "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7" },
            { (const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07", 8, "8a851ff82ee7048ad09ec3847f1ddf44944104d2cbd17ef4e3db22c6785a0d45" },
            { (const uint8_t *)"aaaaaaaaaa", 10, "bf2cb58a68f684d95a3b78ef8f661c9a4e5b09e82cc8f9cc88cce90528caeb27" },
        };

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_sha256_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx2(states, blocks);

        uint8_t digest_avx2[32];
        for (int i = 0; i < 8; i++) {
            sha256_state_to_digest(states_data[i], digest_avx2);
            char name[80];
            snprintf(name, sizeof(name), "7.2 sha256_compress_avx2 8路不同消息 lane%d 与标准值一致", i);
            check(name, cases[i].expected, digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.3  sha256_compress_avx2 — 与标量逐路交叉验证（随机化 state）
     *
     * 测试方法：
     *   - 8 路使用不同的非初始 state（模拟多块消息中间状态）
     *   - 用 AVX2 函数压缩，与标量 sha256_ctx 内部压缩结果对比
     *   - 通过 sha256_update 两次（第一块固定，第二块为测试块）来获取标量参考值
     * ------------------------------------------------------------------ */
    {
        /* 用 8 个不同的 33 字节消息（公钥格式）作为第一块内容，
         * 通过 sha256_update 处理第一块后，ctx.state 即为中间 state，
         * 再处理第二块（全零 64 字节），得到最终 state 作为参考 */
        static const uint8_t first_msgs[8][33] = {
            { 0x02,0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,
              0x07,0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98 },
            { 0x03,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20 },
            { 0x02,0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,
              0xf0,0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0x03,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
              0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99 },
            { 0x02,0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,
              0x01,0x11,0x21,0x31,0x41,0x51,0x61,0x71,0x81,0x91,0xa1,0xb1,0xc1,0xd1,0xe1,0xf1,0x01 },
            { 0x03,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,
              0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
              0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01 },
            { 0x03,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe },
        };

        /* 第二块：全零 64 字节（作为 AVX2 压缩的输入 block） */
        uint8_t zero_block[64];
        memset(zero_block, 0, 64);

        /* 获取每路中间 state：处理完 33 字节后，ctx.buf 中有 33 字节待处理，
         * 此时 state 还是初始值，需要再 update 31 字节凑满一块触发压缩。
         * 更简单的方法：直接用 sha256_ctx 处理 33+31=64 字节（恰好一块），
         * 然后读取 ctx.state 作为中间 state。
         * 但 sha256_ctx 的 state 字段是内部的，我们通过构造 padded block 来模拟。
         *
         * 实际上，最简单的交叉验证方法：
         *   对于每路 i，构造 padded block = sha256_33(first_msgs[i]) 的内部 block，
         *   即：first_msgs[i] (33字节) + 0x80 + 22字节零 + 8字节位长(264 bits)
         *   这恰好是 sha256_33 内部处理的第一个（也是唯一一个）block。
         *   AVX2 压缩后的 state 转为摘要，应等于 sha256_33(first_msgs[i])。
         */
        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][8];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_sha256_padded_block(first_msgs[i], 33, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx2(states, blocks);

        /* 参考值：sha256_33() 的输出 */
        uint8_t digest_avx2[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 8; i++) {
            sha256_state_to_digest(states_data[i], digest_avx2);
            sha256_33(first_msgs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.3 sha256_compress_avx2 8路33字节公钥 lane%d 与sha256_33一致", i);
            check(name, ref_hex, digest_avx2, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.4  ripemd160_compress_avx2 — 8路相同消息，结果与标量 ripemd160() 一致
     * ------------------------------------------------------------------ */
    {
        /* "abc" 的 RIPEMD160 padded block */
        uint8_t block_abc[64];
        make_ripemd160_padded_block((const uint8_t *)"abc", 3, block_abc);

        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];
        for (int i = 0; i < 8; i++) {
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        for (int lane = 0; lane < 8; lane++) {
            rmd160_state_to_digest(states_data[lane], digest_avx2);
            char name[64];
            snprintf(name, sizeof(name), "7.4 ripemd160_compress_avx2(\"abc\") lane%d 与标准值一致", lane);
            check(name,
                  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
                  digest_avx2, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.5  ripemd160_compress_avx2 — 8路不同消息，每路结果与标量一致
     * ------------------------------------------------------------------ */
    {
        static const struct { const uint8_t *msg; size_t len; const char *expected; } cases[8] = {
            { (const uint8_t *)"",          0,  "9c1185a5c5e9fc54612808977ee8f548b2258d31" },
            { (const uint8_t *)"abc",       3,  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc" },
            { (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26, "f71c27109c692c1b56bbdceb5b9d2865b3708dbc" },
            { (const uint8_t *)"hello",     5,  "108f07b8382412612c048d07d13f814118445acd" },
            { (const uint8_t *)"world",     5,  "9b2a277a3e3b3a31b3114ca2d73be6d493d037f9" },
            { (const uint8_t *)"\x00",      1,  "c81b94933420221a7ac004a90242d8b1d3e5070d" },
            { (const uint8_t *)"\xff",      1,  "f7f5b1e2d1b9e3b3e3b3e3b3e3b3e3b3e3b3e3b3" /* 占位，下面用自一致性 */ },
            { (const uint8_t *)"0123456789", 10, "9a1c58e8f2f9b3e3b3e3b3e3b3e3b3e3b3e3b3e3" /* 占位 */ },
        };
        /* 注意：部分期望值用占位符，实际用自一致性验证（与通用 ripemd160 对比） */

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_ripemd160_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 8; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx2);
            /* 用通用接口计算参考值（自一致性，不依赖硬编码期望值） */
            ripemd160(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.5 ripemd160_compress_avx2 8路不同消息 lane%d 与通用ripemd160一致", i);
            check(name, ref_hex, digest_avx2, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 7.6  ripemd160_compress_avx2 — 8路32字节消息（ripemd160_32 场景）
     *
     * 测试方法：
     *   - 8 路使用不同的 32 字节输入（模拟 sha256 输出）
     *   - AVX2 压缩结果与 ripemd160_32() 对比
     * ------------------------------------------------------------------ */
    {
        static const uint8_t inputs[8][32] = {
            /* sha256(G_compressed) */
            { 0x0b,0x7c,0x28,0xc9,0xb7,0x29,0x0c,0x98,0xd7,0x43,0x8e,0x70,0xb3,0xd3,0xf7,0xc8,
              0x48,0xfb,0xd7,0xd1,0xdc,0x19,0x4f,0xf8,0x3f,0x4f,0x7c,0xc9,0xb1,0x37,0x8e,0x98 },
            /* 全零 */
            { 0 },
            /* 全0xFF */
            { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
            /* 递增 0x00~0x1F */
            { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f },
            /* 递减 0xFF~0xE0 */
            { 0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,
              0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            /* 交替 0xAA/0x55 */
            { 0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,
              0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55 },
            /* 随机样本1 */
            { 0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
              0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            /* 随机样本2 */
            { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,
              0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff },
        };

        uint8_t padded_blocks[8][64];
        uint32_t states_data[8][5];
        uint32_t *states[8];
        const uint8_t *blocks[8];

        for (int i = 0; i < 8; i++) {
            make_ripemd160_padded_block(inputs[i], 32, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx2(states, blocks);

        uint8_t digest_avx2[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 8; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx2);
            ripemd160_32(inputs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "7.6 ripemd160_compress_avx2 8路32字节输入 lane%d 与ripemd160_32一致", i);
            check(name, ref_hex, digest_avx2, 20);
        }
    }
}

/* ===================== hash160_8way 测试 ===================== */

/*
 * 测试 hash160_8way_compressed 和 hash160_8way_uncompressed
 * 验证 8路并行结果与标量 pubkey_bytes_to_hash160 一致
 */
static void test_hash160_8way(void) {
    printf("\n=== hash160_8way 8路并行 hash160 测试 ===\n");

    /* 已知公钥：G点（私钥 k=1）的压缩和非压缩公钥 */
    static const uint8_t G_comp[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
    };
    static const uint8_t G_uncomp[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };

    /* ------------------------------------------------------------------ */
    /* 8.1  hash160_8way_compressed — 8路相同输入（G点压缩公钥）
     *      验证 8 路输出均与标量 pubkey_bytes_to_hash160 一致
     * ------------------------------------------------------------------ */
    {
        const uint8_t *comp_ptrs[8];
        for (int i = 0; i < 8; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        /* 标量参考值 */
        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_comp, 33, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 8; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "8.1 hash160_8way_compressed(G点) lane%d 与标量一致", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.2  hash160_8way_uncompressed — 8路相同输入（G点非压缩公钥）
     *      验证 8 路输出均与标量 pubkey_bytes_to_hash160 一致
     * ------------------------------------------------------------------ */
    {
        const uint8_t *uncomp_ptrs[8];
        for (int i = 0; i < 8; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        /* 标量参考值 */
        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_uncomp, 65, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 8; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "8.2 hash160_8way_uncompressed(G点) lane%d 与标量一致", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.3  hash160_8way_compressed — 8路不同输入（k=1..8 的压缩公钥）
     *      逐 lane 与标量 privkey_to_hash160 交叉验证
     * ------------------------------------------------------------------ */
    {
        /* 构造 k=1..8 的压缩公钥 */
        uint8_t comp_bufs[8][33];
        const uint8_t *comp_ptrs[8];
        uint8_t ref_hash160s[8][20];

        for (int i = 0; i < 8; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);
            privkey_to_compressed_bytes(privkey, comp_bufs[i]);
            comp_ptrs[i] = comp_bufs[i];
            privkey_to_hash160(privkey, ref_hash160s[i], NULL);
        }

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        for (int lane = 0; lane < 8; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "8.3 hash160_8way_compressed(k=%d) lane%d 与标量一致", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.4  hash160_8way_uncompressed — 8路不同输入（k=1..8 的非压缩公钥）
     *      逐 lane 与标量 privkey_to_hash160 交叉验证
     * ------------------------------------------------------------------ */
    {
        /* 构造 k=1..8 的非压缩公钥 */
        uint8_t uncomp_bufs[8][65];
        const uint8_t *uncomp_ptrs[8];
        uint8_t ref_hash160s[8][20];

        for (int i = 0; i < 8; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);

            secp256k1_pubkey pubkey;
            size_t len = 65;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
            secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bufs[i], &len,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);
            uncomp_ptrs[i] = uncomp_bufs[i];
            privkey_to_hash160(privkey, NULL, ref_hash160s[i]);
        }

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        for (int lane = 0; lane < 8; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "8.4 hash160_8way_uncompressed(k=%d) lane%d 与标量一致", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 8.5  已知向量：hash160_8way_compressed(G点) lane0 与已知 hash160 一致
     *      G点压缩公钥 hash160 = 751e76e8199196d454941c45d1b3a323f1433bd6
     * ------------------------------------------------------------------ */
    {
        const uint8_t *comp_ptrs[8];
        for (int i = 0; i < 8; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[8][20];
        hash160_8way_compressed(comp_ptrs, hash160s);

        check("8.5 hash160_8way_compressed(G点) 与已知向量一致",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[0], 20);
    }

    /* ------------------------------------------------------------------ */
    /* 8.6  已知向量：hash160_8way_uncompressed(G点) lane0 与已知 hash160 一致
     *      G点非压缩公钥 hash160 = 91b24bf9f5288532960ac687abb035127b1d28a5
     * ------------------------------------------------------------------ */
    {
        const uint8_t *uncomp_ptrs[8];
        for (int i = 0; i < 8; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[8][20];
        hash160_8way_uncompressed(uncomp_ptrs, hash160s);

        check("8.6 hash160_8way_uncompressed(G点) 与已知向量一致",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[0], 20);
    }
}

#endif /* __AVX2__ */

/* ===================== AVX-512 压缩函数测试 ===================== */

#ifdef __AVX512F__

__attribute__((target("avx512f")))
static void test_avx512_compress(void) {
    printf("\n=== AVX-512 压缩函数测试（sha256_compress_avx512 / ripemd160_compress_avx512）===\n");

    /* ------------------------------------------------------------------ */
    /* 10.1  sha256_compress_avx512 — 16路相同消息，结果与标量 sha256() 一致 */
    {
        uint8_t block_abc[64];
        make_sha256_padded_block((const uint8_t *)"abc", 3, block_abc);

        uint32_t states_data[16][8];
        uint32_t *states[16];
        const uint8_t *blocks[16];
        for (int i = 0; i < 16; i++) {
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        sha256_compress_avx512(states, blocks);

        uint8_t digest_avx512[32];
        for (int lane = 0; lane < 16; lane++) {
            sha256_state_to_digest(states_data[lane], digest_avx512);
            char name[80];
            snprintf(name, sizeof(name), "10.1 sha256_compress_avx512(\"abc\") lane%d 与标准值一致", lane);
            check(name,
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                  digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.2  sha256_compress_avx512 — 16路不同消息，每路结果与标量 sha256() 一致 */
    {
        static const struct { const uint8_t *msg; size_t len; } cases[16] = {
            { (const uint8_t *)"",           0  },
            { (const uint8_t *)"abc",        3  },
            { (const uint8_t *)"\x00",       1  },
            { (const uint8_t *)"\xff",       1  },
            { (const uint8_t *)"hello",      5  },
            { (const uint8_t *)"world",      5  },
            { (const uint8_t *)"\x00\x01\x02\x03\x04\x05\x06\x07", 8 },
            { (const uint8_t *)"aaaaaaaaaa", 10 },
            { (const uint8_t *)"bitcoin",    7  },
            { (const uint8_t *)"secp256k1",  9  },
            { (const uint8_t *)"ripemd160",  9  },
            { (const uint8_t *)"sha256",     6  },
            { (const uint8_t *)"avx512",     6  },
            { (const uint8_t *)"keysearch",  9  },
            { (const uint8_t *)"test16way",  9  },
            { (const uint8_t *)"lane15msg",  9  },
        };

        uint8_t padded_blocks[16][64];
        uint32_t states_data[16][8];
        uint32_t *states[16];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_sha256_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx512(states, blocks);

        uint8_t digest_avx512[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 16; i++) {
            sha256_state_to_digest(states_data[i], digest_avx512);
            sha256(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.2 sha256_compress_avx512 16路不同消息 lane%d 与标量一致", i);
            check(name, ref_hex, digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.3  sha256_compress_avx512 — 16路33字节公钥，与 sha256_33 交叉验证 */
    {
        static const uint8_t msgs[16][33] = {
            { 0x02,0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,
              0x07,0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98 },
            { 0x03,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20 },
            { 0x02,0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,
              0xf0,0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0x03,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
              0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99 },
            { 0x02,0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,
              0x01,0x11,0x21,0x31,0x41,0x51,0x61,0x71,0x81,0x91,0xa1,0xb1,0xc1,0xd1,0xe1,0xf1,0x01 },
            { 0x03,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,
              0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
              0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01 },
            { 0x03,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe },
            { 0x02,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,
              0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00 },
            { 0x03,0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
              0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            { 0x02,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,0x4a,0x5b,0x6c,0x7d,0x8e,
              0x9f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,0x4a,0x5b,0x6c,0x7d,0x8e,0x9f },
            { 0x03,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,
              0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0 },
            { 0x02,0x55,0x44,0x33,0x22,0x11,0x00,0xff,0xee,0xdd,0xcc,0xbb,0xaa,0x99,0x88,0x77,
              0x66,0x55,0x44,0x33,0x22,0x11,0x00,0xff,0xee,0xdd,0xcc,0xbb,0xaa,0x99,0x88,0x77,0x66 },
            { 0x03,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
              0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef },
            { 0x02,0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b,0x3c,0x2d,0x1e,
              0x0f,0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f },
            { 0x03,0x80,0x70,0x60,0x50,0x40,0x30,0x20,0x10,0x08,0x07,0x06,0x05,0x04,0x03,0x02,
              0x01,0x80,0x70,0x60,0x50,0x40,0x30,0x20,0x10,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01 },
        };

        uint8_t padded_blocks[16][64];
        uint32_t states_data[16][8];
        uint32_t *states[16];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_sha256_padded_block(msgs[i], 33, padded_blocks[i]);
            memcpy(states_data[i], SHA256_INIT, 32);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        sha256_compress_avx512(states, blocks);

        uint8_t digest_avx512[32];
        uint8_t digest_ref[32];
        char ref_hex[65];
        for (int i = 0; i < 16; i++) {
            sha256_state_to_digest(states_data[i], digest_avx512);
            sha256_33(msgs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 32, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.3 sha256_compress_avx512 16路33字节公钥 lane%d 与sha256_33一致", i);
            check(name, ref_hex, digest_avx512, 32);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.4  ripemd160_compress_avx512 — 16路相同消息，结果与标量 ripemd160() 一致 */
    {
        uint8_t block_abc[64];
        make_ripemd160_padded_block((const uint8_t *)"abc", 3, block_abc);

        uint32_t states_data[16][5];
        uint32_t *states[16];
        const uint8_t *blocks[16];
        for (int i = 0; i < 16; i++) {
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = block_abc;
        }

        ripemd160_compress_avx512(states, blocks);

        uint8_t digest_avx512[20];
        for (int lane = 0; lane < 16; lane++) {
            rmd160_state_to_digest(states_data[lane], digest_avx512);
            char name[80];
            snprintf(name, sizeof(name), "10.4 ripemd160_compress_avx512(\"abc\") lane%d 与标准值一致", lane);
            check(name,
                  "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc",
                  digest_avx512, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.5  ripemd160_compress_avx512 — 16路不同消息，每路结果与标量一致 */
    {
        static const struct { const uint8_t *msg; size_t len; } cases[16] = {
            { (const uint8_t *)"",           0  },
            { (const uint8_t *)"abc",        3  },
            { (const uint8_t *)"abcdefghijklmnopqrstuvwxyz", 26 },
            { (const uint8_t *)"hello",      5  },
            { (const uint8_t *)"world",      5  },
            { (const uint8_t *)"\x00",       1  },
            { (const uint8_t *)"\xff",       1  },
            { (const uint8_t *)"0123456789", 10 },
            { (const uint8_t *)"bitcoin",    7  },
            { (const uint8_t *)"secp256k1",  9  },
            { (const uint8_t *)"ripemd160",  9  },
            { (const uint8_t *)"sha256",     6  },
            { (const uint8_t *)"avx512",     6  },
            { (const uint8_t *)"keysearch",  9  },
            { (const uint8_t *)"test16way",  9  },
            { (const uint8_t *)"lane15msg",  9  },
        };

        uint8_t padded_blocks[16][64];
        uint32_t states_data[16][5];
        uint32_t *states[16];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_ripemd160_padded_block(cases[i].msg, cases[i].len, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx512(states, blocks);

        uint8_t digest_avx512[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 16; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx512);
            ripemd160(cases[i].msg, cases[i].len, digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.5 ripemd160_compress_avx512 16路不同消息 lane%d 与通用ripemd160一致", i);
            check(name, ref_hex, digest_avx512, 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 10.6  ripemd160_compress_avx512 — 16路32字节消息（ripemd160_32 场景） */
    {
        static const uint8_t inputs[16][32] = {
            { 0x0b,0x7c,0x28,0xc9,0xb7,0x29,0x0c,0x98,0xd7,0x43,0x8e,0x70,0xb3,0xd3,0xf7,0xc8,
              0x48,0xfb,0xd7,0xd1,0xdc,0x19,0x4f,0xf8,0x3f,0x4f,0x7c,0xc9,0xb1,0x37,0x8e,0x98 },
            { 0 },
            { 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
              0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff },
            { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
              0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f },
            { 0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,
              0xef,0xee,0xed,0xec,0xeb,0xea,0xe9,0xe8,0xe7,0xe6,0xe5,0xe4,0xe3,0xe2,0xe1,0xe0 },
            { 0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,
              0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55,0xaa,0x55 },
            { 0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,
              0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88 },
            { 0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10,
              0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff,0x00,0xff },
            { 0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,
              0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00 },
            { 0xa1,0xb2,0xc3,0xd4,0xe5,0xf6,0x07,0x18,0x29,0x3a,0x4b,0x5c,0x6d,0x7e,0x8f,0x90,
              0xa1,0xb2,0xc3,0xd4,0xe5,0xf6,0x07,0x18,0x29,0x3a,0x4b,0x5c,0x6d,0x7e,0x8f,0x90 },
            { 0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,0x01,
              0x10,0x20,0x30,0x40,0x50,0x60,0x70,0x80,0x90,0xa0,0xb0,0xc0,0xd0,0xe0,0xf0,0x01 },
            { 0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39,
              0x5a,0x4b,0x3c,0x2d,0x1e,0x0f,0xa0,0xb1,0xc2,0xd3,0xe4,0xf5,0x06,0x17,0x28,0x39 },
            { 0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,
              0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0x9b,0x9c,0x9d,0x9e,0x9f },
            { 0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x7b,0x7c,0x7d,0x7e,0x7f,
              0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x6b,0x6c,0x6d,0x6e,0x6f },
            { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,
              0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80 },
            { 0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,
              0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f },
        };

        uint8_t padded_blocks[16][64];
        uint32_t states_data[16][5];
        uint32_t *states[16];
        const uint8_t *blocks[16];

        for (int i = 0; i < 16; i++) {
            make_ripemd160_padded_block(inputs[i], 32, padded_blocks[i]);
            memcpy(states_data[i], RMD160_INIT, 20);
            states[i] = states_data[i];
            blocks[i] = padded_blocks[i];
        }

        ripemd160_compress_avx512(states, blocks);

        uint8_t digest_avx512[20];
        uint8_t digest_ref[20];
        char ref_hex[41];
        for (int i = 0; i < 16; i++) {
            rmd160_state_to_digest(states_data[i], digest_avx512);
            ripemd160_32(inputs[i], digest_ref);
            bytes_to_hex_helper(digest_ref, 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name), "10.6 ripemd160_compress_avx512 16路32字节输入 lane%d 与ripemd160_32一致", i);
            check(name, ref_hex, digest_avx512, 20);
        }
    }
}

/* ===================== hash160_16way 测试 ===================== */

__attribute__((target("avx512f")))
static void test_hash160_16way(void) {
    printf("\n=== hash160_16way 16路并行 hash160 测试 ===\n");

    static const uint8_t G_comp[33] = {
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
        0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
        0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
    };
    static const uint8_t G_uncomp[65] = {
        0x04,
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
        0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
        0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
        0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
        0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
        0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
        0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
        0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
    };

    /* ------------------------------------------------------------------ */
    /* 11.1  hash160_16way_compressed — 16路相同输入（G点压缩公钥） */
    {
        const uint8_t *comp_ptrs[16];
        for (int i = 0; i < 16; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[16][20];
        hash160_16way_compressed(comp_ptrs, hash160s);

        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_comp, 33, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 16; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "11.1 hash160_16way_compressed(G点) lane%d 与标量一致", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.2  hash160_16way_uncompressed — 16路相同输入（G点非压缩公钥） */
    {
        const uint8_t *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed(uncomp_ptrs, hash160s);

        uint8_t ref[20];
        pubkey_bytes_to_hash160(G_uncomp, 65, ref);
        char ref_hex[41];
        bytes_to_hex_helper(ref, 20, ref_hex);

        for (int lane = 0; lane < 16; lane++) {
            char name[80];
            snprintf(name, sizeof(name),
                     "11.2 hash160_16way_uncompressed(G点) lane%d 与标量一致", lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.3  hash160_16way_compressed — 16路不同输入（k=1..16 的压缩公钥） */
    {
        uint8_t comp_bufs[16][33];
        const uint8_t *comp_ptrs[16];
        uint8_t ref_hash160s[16][20];

        for (int i = 0; i < 16; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);
            privkey_to_compressed_bytes(privkey, comp_bufs[i]);
            comp_ptrs[i] = comp_bufs[i];
            privkey_to_hash160(privkey, ref_hash160s[i], NULL);
        }

        uint8_t hash160s[16][20];
        hash160_16way_compressed(comp_ptrs, hash160s);

        for (int lane = 0; lane < 16; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "11.3 hash160_16way_compressed(k=%d) lane%d 与标量一致", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.4  hash160_16way_uncompressed — 16路不同输入（k=1..16 的非压缩公钥） */
    {
        uint8_t uncomp_bufs[16][65];
        const uint8_t *uncomp_ptrs[16];
        uint8_t ref_hash160s[16][20];

        for (int i = 0; i < 16; i++) {
            uint8_t privkey[32] = {0};
            privkey[31] = (uint8_t)(i + 1);

            secp256k1_pubkey pubkey;
            size_t len = 65;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey);
            secp256k1_ec_pubkey_serialize(secp_ctx, uncomp_bufs[i], &len,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);
            uncomp_ptrs[i] = uncomp_bufs[i];
            privkey_to_hash160(privkey, NULL, ref_hash160s[i]);
        }

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed(uncomp_ptrs, hash160s);

        for (int lane = 0; lane < 16; lane++) {
            char ref_hex[41];
            bytes_to_hex_helper(ref_hash160s[lane], 20, ref_hex);
            char name[80];
            snprintf(name, sizeof(name),
                     "11.4 hash160_16way_uncompressed(k=%d) lane%d 与标量一致", lane + 1, lane);
            check(name, ref_hex, hash160s[lane], 20);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 11.5  已知向量：hash160_16way_compressed(G点) lane0 与已知 hash160 一致 */
    {
        const uint8_t *comp_ptrs[16];
        for (int i = 0; i < 16; i++) comp_ptrs[i] = G_comp;

        uint8_t hash160s[16][20];
        hash160_16way_compressed(comp_ptrs, hash160s);

        check("11.5 hash160_16way_compressed(G点) 与已知向量一致",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[0], 20);
        check("11.5b hash160_16way_compressed(G点) lane15 与已知向量一致",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              hash160s[15], 20);
    }

    /* ------------------------------------------------------------------ */
    /* 11.6  已知向量：hash160_16way_uncompressed(G点) lane0 与已知 hash160 一致 */
    {
        const uint8_t *uncomp_ptrs[16];
        for (int i = 0; i < 16; i++) uncomp_ptrs[i] = G_uncomp;

        uint8_t hash160s[16][20];
        hash160_16way_uncompressed(uncomp_ptrs, hash160s);

        check("11.6 hash160_16way_uncompressed(G点) 与已知向量一致",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[0], 20);
        check("11.6b hash160_16way_uncompressed(G点) lane15 与已知向量一致",
              "91b24bf9f5288532960ac687abb035127b1d28a5",
              hash160s[15], 20);
    }
}

/* ===================== ht_contains_16way 测试 ===================== */

__attribute__((target("avx512f")))
static void test_ht_contains_16way_func(void) {
    printf("\n=== ht_contains_16way AVX-512 批量查表测试 ===\n");

    if (ht_init(128) != 0) {
        printf("  [FAIL] 12.0 ht_init 失败\n");
        fail_count++;
        return;
    }

    /* 准备16个已知 hash160 */
    uint8_t known[16][20];
    for (int i = 0; i < 16; i++) {
        memset(known[i], 0, 20);
        known[i][0] = (uint8_t)(0x10 + i);
        known[i][1] = (uint8_t)(0x20 + i);
        known[i][2] = (uint8_t)(0x30 + i);
        known[i][3] = (uint8_t)(0x40 + i);
        known[i][19] = (uint8_t)(i + 1);
        ht_insert(known[i]);
    }

    /* 准备16个未插入的 hash160 */
    uint8_t unknown[16][20];
    for (int i = 0; i < 16; i++) {
        memset(unknown[i], 0, 20);
        unknown[i][0] = (uint8_t)(0xf0 + i);
        unknown[i][1] = (uint8_t)(0xe0 + i);
        unknown[i][2] = (uint8_t)(0xd0 + i);
        unknown[i][3] = (uint8_t)(0xc0 + i);
        unknown[i][19] = (uint8_t)(i + 0x80);
    }

    /* 12.1 16路全命中 */
    const uint8_t *ptrs_all[16];
    for (int i = 0; i < 16; i++) ptrs_all[i] = known[i];
    uint16_t mask = ht_contains_16way(ptrs_all);
    if (mask == 0xffff) {
        printf("  [PASS] 12.1 16路全命中，掩码=0xffff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.1 16路全命中，期望掩码=0xffff，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.2 16路全未命中 */
    const uint8_t *ptrs_none[16];
    for (int i = 0; i < 16; i++) ptrs_none[i] = unknown[i];
    mask = ht_contains_16way(ptrs_none);
    if (mask == 0x0000) {
        printf("  [PASS] 12.2 16路全未命中，掩码=0x0000\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.2 16路全未命中，期望掩码=0x0000，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.3 部分命中：偶数 lane 命中（0,2,4,6,8,10,12,14），期望掩码=0x5555 */
    const uint8_t *ptrs_mix[16];
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i % 2 == 0) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x5555) {
        printf("  [PASS] 12.3 部分命中（偶数lane），掩码=0x5555\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.3 部分命中（偶数lane），期望掩码=0x5555，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.4 部分命中：奇数 lane 命中（1,3,5,7,9,11,13,15），期望掩码=0xaaaa */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i % 2 == 1) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0xaaaa) {
        printf("  [PASS] 12.4 部分命中（奇数lane），掩码=0xaaaa\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.4 部分命中（奇数lane），期望掩码=0xaaaa，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.5 仅 lane0 命中，期望掩码=0x0001 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i == 0) ? known[0] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x0001) {
        printf("  [PASS] 12.5 仅 lane0 命中，掩码=0x0001\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.5 仅 lane0 命中，期望掩码=0x0001，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.6 仅 lane15 命中，期望掩码=0x8000 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i == 15) ? known[15] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x8000) {
        printf("  [PASS] 12.6 仅 lane15 命中，掩码=0x8000\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.6 仅 lane15 命中，期望掩码=0x8000，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.7 前8路命中，后8路未命中，期望掩码=0x00ff */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i < 8) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0x00ff) {
        printf("  [PASS] 12.7 前8路命中，掩码=0x00ff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.7 前8路命中，期望掩码=0x00ff，实际=0x%04x\n", mask);
        fail_count++;
    }

    /* 12.8 后8路命中，前8路未命中，期望掩码=0xff00 */
    for (int i = 0; i < 16; i++) {
        ptrs_mix[i] = (i >= 8) ? known[i] : unknown[i];
    }
    mask = ht_contains_16way(ptrs_mix);
    if (mask == 0xff00) {
        printf("  [PASS] 12.8 后8路命中，掩码=0xff00\n");
        pass_count++;
    } else {
        printf("  [FAIL] 12.8 后8路命中，期望掩码=0xff00，实际=0x%04x\n", mask);
        fail_count++;
    }

    ht_free();
}

/* AVX-512 函数前向声明（供测试调用） */
void sha256_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16]);
void ripemd160_compress_avx512(uint32_t *states[16], const uint8_t *blocks[16]);

#endif /* __AVX512F__ */

/* ===================== 开放寻址哈希表测试 ===================== */

static void test_ht_openaddr(void) {
    printf("\n=== 开放寻址哈希表测试 ===\n");

    /* 初始化哈希表（16个槽位，足够测试用） */
    if (ht_init(16) != 0) {
        printf("  [FAIL] 8.0 ht_init 失败\n");
        fail_count++;
        return;
    }

    /* 8.1 插入已知 hash160，验证 ht_contains 返回1 */
    uint8_t h1[20] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,
                      0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14};
    uint8_t h2[20] = {0xaa,0xbb,0xcc,0xdd,0xee,0xff,0x00,0x11,0x22,0x33,
                      0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd};
    ht_insert(h1);
    ht_insert(h2);

    if (ht_contains(h1) == 1) {
        printf("  [PASS] 8.1 ht_contains(h1) 命中\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.1 ht_contains(h1) 应命中但未命中\n");
        fail_count++;
    }

    if (ht_contains(h2) == 1) {
        printf("  [PASS] 8.2 ht_contains(h2) 命中\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.2 ht_contains(h2) 应命中但未命中\n");
        fail_count++;
    }

    /* 8.3 查找未插入的 hash160，验证返回0 */
    uint8_t h3[20] = {0xde,0xad,0xbe,0xef,0x00,0x00,0x00,0x00,0x00,0x00,
                      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    if (ht_contains(h3) == 0) {
        printf("  [PASS] 8.3 ht_contains(h3) 未命中（正确）\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.3 ht_contains(h3) 应未命中但命中\n");
        fail_count++;
    }

    /* 8.4 fp==0 边界：hash160 前4字节全零 */
    uint8_t h_fp0[20] = {0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,
                         0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    ht_insert(h_fp0);
    if (ht_contains(h_fp0) == 1) {
        printf("  [PASS] 8.4 fp==0 边界：ht_contains(h_fp0) 命中\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.4 fp==0 边界：ht_contains(h_fp0) 应命中但未命中\n");
        fail_count++;
    }

    /* 8.5 大量插入（负载因子接近0.5），重新初始化更大的表 */
    ht_free();
    if (ht_init(256) != 0) {
        printf("  [FAIL] 8.5 ht_init(256) 失败\n");
        fail_count++;
        return;
    }

    int n = 100; /* 插入100条，负载因子 100/256 ≈ 0.39 */
    uint8_t keys[100][20];
    for (int i = 0; i < n; i++) {
        memset(keys[i], 0, 20);
        keys[i][0] = (uint8_t)(i >> 8);
        keys[i][1] = (uint8_t)(i & 0xff);
        keys[i][19] = (uint8_t)(i * 7 + 3); /* 增加差异性 */
        ht_insert(keys[i]);
    }

    int all_found = 1;
    for (int i = 0; i < n; i++) {
        if (!ht_contains(keys[i])) {
            printf("  [FAIL] 8.5 大量插入：第%d条未命中\n", i);
            all_found = 0;
            fail_count++;
            break;
        }
    }
    if (all_found) {
        printf("  [PASS] 8.5 大量插入（100条）全部命中\n");
        pass_count++;
    }

    /* 8.6 验证未插入的 key 不会误命中 */
    uint8_t h_miss[20] = {0xff,0xfe,0xfd,0xfc,0xfb,0xfa,0xf9,0xf8,0xf7,0xf6,
                          0xf5,0xf4,0xf3,0xf2,0xf1,0xf0,0xef,0xee,0xed,0xec};
    if (ht_contains(h_miss) == 0) {
        printf("  [PASS] 8.6 未插入 key 不误命中\n");
        pass_count++;
    } else {
        printf("  [FAIL] 8.6 未插入 key 误命中\n");
        fail_count++;
    }

    ht_free();
}

#ifdef __AVX2__
static void test_ht_contains_8way_func(void) {
    printf("\n=== ht_contains_8way AVX2 批量查表测试 ===\n");

    /* 初始化哈希表 */
    if (ht_init(64) != 0) {
        printf("  [FAIL] 9.0 ht_init 失败\n");
        fail_count++;
        return;
    }

    /* 准备8个已知 hash160 */
    uint8_t known[8][20];
    for (int i = 0; i < 8; i++) {
        memset(known[i], 0, 20);
        known[i][0] = (uint8_t)(0x10 + i);
        known[i][1] = (uint8_t)(0x20 + i);
        known[i][2] = (uint8_t)(0x30 + i);
        known[i][3] = (uint8_t)(0x40 + i);
        known[i][19] = (uint8_t)(i + 1);
        ht_insert(known[i]);
    }

    /* 准备8个未插入的 hash160 */
    uint8_t unknown[8][20];
    for (int i = 0; i < 8; i++) {
        memset(unknown[i], 0, 20);
        unknown[i][0] = (uint8_t)(0xf0 + i);
        unknown[i][1] = (uint8_t)(0xe0 + i);
        unknown[i][2] = (uint8_t)(0xd0 + i);
        unknown[i][3] = (uint8_t)(0xc0 + i);
        unknown[i][19] = (uint8_t)(i + 0x80);
    }

    /* 9.1 8路全命中 */
    const uint8_t *ptrs_all[8];
    for (int i = 0; i < 8; i++) ptrs_all[i] = known[i];
    uint8_t mask = ht_contains_8way(ptrs_all);
    if (mask == 0xff) {
        printf("  [PASS] 9.1 8路全命中，掩码=0xff\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.1 8路全命中，期望掩码=0xff，实际=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.2 8路全未命中 */
    const uint8_t *ptrs_none[8];
    for (int i = 0; i < 8; i++) ptrs_none[i] = unknown[i];
    mask = ht_contains_8way(ptrs_none);
    if (mask == 0x00) {
        printf("  [PASS] 9.2 8路全未命中，掩码=0x00\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.2 8路全未命中，期望掩码=0x00，实际=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.3 部分命中：偶数 lane 命中（0,2,4,6），期望掩码=0x55 */
    const uint8_t *ptrs_mix[8];
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i % 2 == 0) ? known[i] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x55) {
        printf("  [PASS] 9.3 部分命中（偶数lane），掩码=0x55\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.3 部分命中（偶数lane），期望掩码=0x55，实际=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.4 部分命中：奇数 lane 命中（1,3,5,7），期望掩码=0xaa */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i % 2 == 1) ? known[i] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0xaa) {
        printf("  [PASS] 9.4 部分命中（奇数lane），掩码=0xaa\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.4 部分命中（奇数lane），期望掩码=0xaa，实际=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.5 仅 lane0 命中，期望掩码=0x01 */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i == 0) ? known[0] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x01) {
        printf("  [PASS] 9.5 仅 lane0 命中，掩码=0x01\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.5 仅 lane0 命中，期望掩码=0x01，实际=0x%02x\n", mask);
        fail_count++;
    }

    /* 9.6 仅 lane7 命中，期望掩码=0x80 */
    for (int i = 0; i < 8; i++) {
        ptrs_mix[i] = (i == 7) ? known[7] : unknown[i];
    }
    mask = ht_contains_8way(ptrs_mix);
    if (mask == 0x80) {
        printf("  [PASS] 9.6 仅 lane7 命中，掩码=0x80\n");
        pass_count++;
    } else {
        printf("  [FAIL] 9.6 仅 lane7 命中，期望掩码=0x80，实际=0x%02x\n", mask);
        fail_count++;
    }

    ht_free();
}
#endif /* __AVX2__ */

/* ===================== 专用接口测试：sha256_33 / sha256_65 / ripemd160_32 ===================== */

static void test_specialized_interfaces(void) {
    printf("\n=== 专用接口测试：sha256_33 / sha256_65 / ripemd160_32 ===\n");

    uint8_t digest_spec[32];
    uint8_t digest_ref[32];
    uint8_t rmd_spec[20];
    uint8_t rmd_ref[20];
    char ref_hex[65];

    /* ------------------------------------------------------------------ */
    /* 6.1  sha256_33 — G点压缩公钥已知向量
     *   输入: 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
     *   期望: 0b7c28c9b7290c98d7438e70b3d3f7c848fbd7d1dc194ff83f4f7cc9b1378e98
     *   （hash160 = ripemd160(sha256(G)) = 751e76e8199196d454941c45d1b3a323f1433bd6 已知）
     * ------------------------------------------------------------------ */
    {
        static const uint8_t G_comp[33] = {
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
            0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
            0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
            0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
        };
        /* 用通用接口计算参考值 */
        sha256(G_comp, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        /* 用专用接口计算 */
        sha256_33(G_comp, digest_spec);
        check("6.1 sha256_33(G点压缩公钥) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.2  sha256_33 — 全零33字节（自一致性） */
    {
        uint8_t zeros33[33];
        memset(zeros33, 0x00, 33);
        sha256(zeros33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(zeros33, digest_spec);
        check("6.2 sha256_33(全零33字节) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.3  sha256_33 — 全0xFF 33字节（自一致性） */
    {
        uint8_t ff33[33];
        memset(ff33, 0xFF, 33);
        sha256(ff33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(ff33, digest_spec);
        check("6.3 sha256_33(全0xFF 33字节) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.4  sha256_33 — 首字节0x02，其余递增（自一致性） */
    {
        uint8_t incr33[33];
        incr33[0] = 0x02;
        for (int i = 1; i < 33; i++) incr33[i] = (uint8_t)(i);
        sha256(incr33, 33, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_33(incr33, digest_spec);
        check("6.4 sha256_33(0x02+递增序列) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.5  sha256_65 — G点非压缩公钥已知向量（自一致性） */
    {
        static const uint8_t G_uncomp[65] = {
            0x04,
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac,
            0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b, 0x07,
            0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9,
            0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98,
            0x48, 0x3a, 0xda, 0x77, 0x26, 0xa3, 0xc4, 0x65,
            0x5d, 0xa4, 0xfb, 0xfc, 0x0e, 0x11, 0x08, 0xa8,
            0xfd, 0x17, 0xb4, 0x48, 0xa6, 0x85, 0x54, 0x19,
            0x9c, 0x47, 0xd0, 0x8f, 0xfb, 0x10, 0xd4, 0xb8
        };
        sha256(G_uncomp, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(G_uncomp, digest_spec);
        check("6.5 sha256_65(G点非压缩公钥) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.6  sha256_65 — 全零65字节（自一致性） */
    {
        uint8_t zeros65[65];
        memset(zeros65, 0x00, 65);
        sha256(zeros65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(zeros65, digest_spec);
        check("6.6 sha256_65(全零65字节) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.7  sha256_65 — 全0xFF 65字节（自一致性） */
    {
        uint8_t ff65[65];
        memset(ff65, 0xFF, 65);
        sha256(ff65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(ff65, digest_spec);
        check("6.7 sha256_65(全0xFF 65字节) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.8  sha256_65 — 首字节0x04，其余递增（自一致性） */
    {
        uint8_t incr65[65];
        incr65[0] = 0x04;
        for (int i = 1; i < 65; i++) incr65[i] = (uint8_t)(i);
        sha256(incr65, 65, digest_ref);
        bytes_to_hex_helper(digest_ref, 32, ref_hex);
        sha256_65(incr65, digest_spec);
        check("6.8 sha256_65(0x04+递增序列) 与通用sha256一致", ref_hex, digest_spec, 32);
    }

    /* ------------------------------------------------------------------ */
    /* 6.9  ripemd160_32 — G点压缩公钥SHA256结果（已知向量）
     *   输入: sha256(G_compressed) = 0b7c28c9b7290c98d7438e70b3d3f7c848fbd7d1dc194ff83f4f7cc9b1378e98
     *   期望: hash160(G) = 751e76e8199196d454941c45d1b3a323f1433bd6
     * ------------------------------------------------------------------ */
    {
        static const uint8_t G_comp[33] = {
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb,
            0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
            0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28,
            0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8, 0x17, 0x98
        };
        /* 先用通用sha256计算G的SHA256（32字节） */
        uint8_t sha256_G[32];
        sha256(G_comp, 33, sha256_G);
        /* 用专用接口计算ripemd160_32 */
        ripemd160_32(sha256_G, rmd_spec);
        check("6.9 ripemd160_32(sha256(G)) 与已知hash160向量一致",
              "751e76e8199196d454941c45d1b3a323f1433bd6",
              rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.10 ripemd160_32 — 全零32字节（自一致性） */
    {
        uint8_t zeros32[32];
        memset(zeros32, 0x00, 32);
        ripemd160(zeros32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(zeros32, rmd_spec);
        check("6.10 ripemd160_32(全零32字节) 与通用ripemd160一致", rmd_ref_hex, rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.11 ripemd160_32 — 全0xFF 32字节（自一致性） */
    {
        uint8_t ff32[32];
        memset(ff32, 0xFF, 32);
        ripemd160(ff32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(ff32, rmd_spec);
        check("6.11 ripemd160_32(全0xFF 32字节) 与通用ripemd160一致", rmd_ref_hex, rmd_spec, 20);
    }

    /* ------------------------------------------------------------------ */
    /* 6.12 ripemd160_32 — 递增序列0x00~0x1F（自一致性） */
    {
        uint8_t incr32[32];
        for (int i = 0; i < 32; i++) incr32[i] = (uint8_t)i;
        ripemd160(incr32, 32, rmd_ref);
        char rmd_ref_hex[41];
        bytes_to_hex_helper(rmd_ref, 20, rmd_ref_hex);
        ripemd160_32(incr32, rmd_spec);
        check("6.12 ripemd160_32(0x00~0x1F递增序列) 与通用ripemd160一致", rmd_ref_hex, rmd_spec, 20);
    }
}

/* ===================== keygen 内部接口正确性测试 ===================== */

#ifndef USE_PUBKEY_API_ONLY

static void test_keygen_internal(void) {
    printf("\n=== keygen 内部接口正确性测试 ===\n");

    /* ------------------------------------------------------------------
     * 5.1  keygen_init_generator：验证 G_affine.infinity == 0
     *      且 G_affine.x / G_affine.y 与已知标准值一致
     * ------------------------------------------------------------------ */
    {
        /* 已知 G 的压缩公钥字节（私钥 k=1 对应的公钥即为 G） */
        static const uint8_t G_compressed_expected[33] = {
            0x02,
            0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
            0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
            0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
            0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
        };

        /* 用 keygen_ge_to_pubkey_bytes 从 G_affine 提取压缩字节 */
        uint8_t G_bytes[33];
        keygen_ge_to_pubkey_bytes(&G_affine, G_bytes, NULL);

        char expected_hex[67];
        bytes_to_hex_helper(G_compressed_expected, 33, expected_hex);
        check("5.1 keygen_init_generator: G 压缩公钥与已知标准值一致",
              expected_hex, G_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.2  keygen_privkey_to_gej + keygen_batch_normalize(n=1)：
     *      验证 k=1 的 Jacobian 点归一化后与直接 serialize 结果一致
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        /* 方法A：keygen_privkey_to_gej + keygen_batch_normalize + keygen_ge_to_pubkey_bytes */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t keygen_bytes[33];
        if (keygen_privkey_to_gej(secp_ctx, privkey1, &gej) == 0) {
            keygen_batch_normalize(&gej, &ge, 1);
            keygen_ge_to_pubkey_bytes(&ge, keygen_bytes, NULL);
        } else {
            memset(keygen_bytes, 0, 33);
        }

        /* 方法B：公开 API serialize */
        uint8_t api_bytes[33];
        privkey_to_compressed_bytes(privkey1, api_bytes);

        char api_hex[67];
        bytes_to_hex_helper(api_bytes, 33, api_hex);
        check("5.2 keygen_privkey_to_gej + batch_normalize(n=1) 与公开 API 一致（k=1）",
              api_hex, keygen_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.3  直接点加法 secp256k1_gej_add_ge + keygen_batch_normalize：
     *      验证 k=1 的 Jacobian 点加 G 后归一化，与直接计算 k=2 一致
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;
        uint8_t privkey2[32] = {0};
        privkey2[31] = 2;

        /* 方法A：gej(k=1) + G_affine -> 归一化 -> 字节 */
        secp256k1_gej gej_k1, gej_k2;
        secp256k1_ge  ge_k2;
        uint8_t incr_bytes[33];
        keygen_privkey_to_gej(secp_ctx, privkey1, &gej_k1);
        secp256k1_gej_add_ge(&gej_k2, &gej_k1, &G_affine);
        keygen_batch_normalize(&gej_k2, &ge_k2, 1);
        keygen_ge_to_pubkey_bytes(&ge_k2, incr_bytes, NULL);

        /* 方法B：直接计算 k=2 */
        uint8_t direct_bytes[33];
        privkey_to_compressed_bytes(privkey2, direct_bytes);

        char direct_hex[67];
        bytes_to_hex_helper(direct_bytes, 33, direct_hex);
        check("5.3 gej_add_ge(k=1, G) + batch_normalize 与直接计算 k=2 一致",
              direct_hex, incr_bytes, 33);
    }

    /* ------------------------------------------------------------------
     * 5.4  keygen_batch_normalize(n=BATCH_SIZE)：
     *      批量归一化 2048 个点，验证每个点与逐个 serialize 结果一致
     * ------------------------------------------------------------------ */
    {
        printf("  [批量归一化 BATCH_SIZE=2048 验证]\n");

        const int N = 2048;
        secp256k1_gej *gej_arr = (secp256k1_gej *)malloc(N * sizeof(secp256k1_gej));
        secp256k1_ge  *ge_arr  = (secp256k1_ge  *)malloc(N * sizeof(secp256k1_ge));

        if (!gej_arr || !ge_arr) {
            printf("  [SKIP] 5.4 内存分配失败，跳过\n");
            free(gej_arr); free(ge_arr);
        } else {
            /* 从 k=1 开始，连续点加 G 生成 2048 个 Jacobian 点 */
            uint8_t privkey[32] = {0};
            privkey[31] = 1;
            secp256k1_gej cur;
            keygen_privkey_to_gej(secp_ctx, privkey, &cur);
            for (int i = 0; i < N; i++) {
                gej_arr[i] = cur;
                secp256k1_gej next;
                secp256k1_gej_add_ge(&next, &cur, &G_affine);
                cur = next;
            }

            /* 批量归一化 */
            keygen_batch_normalize(gej_arr, ge_arr, (size_t)N);

            /* 逐点验证：keygen_ge_to_pubkey_bytes 与公开 API serialize 一致 */
            int all_pass = 1;
            uint8_t privkey_i[32] = {0};
            privkey_i[31] = 1;
            for (int i = 0; i < N; i++) {
                uint8_t keygen_bytes[33];
                uint8_t api_bytes[33];

                keygen_ge_to_pubkey_bytes(&ge_arr[i], keygen_bytes, NULL);
                privkey_to_compressed_bytes(privkey_i, api_bytes);

                if (memcmp(keygen_bytes, api_bytes, 33) != 0) {
                    char kb[67], ab[67];
                    bytes_to_hex_helper(keygen_bytes, 33, kb);
                    bytes_to_hex_helper(api_bytes,    33, ab);
                    printf("  [FAIL] 5.4 第 %d 点不一致:\n"
                           "         keygen: %s\n"
                           "         api:    %s\n", i, kb, ab);
                    all_pass = 0;
                    fail_count++;
                    break;
                }

                /* 私钥递增，对应下一个点 */
                uint8_t tweak[32] = {0};
                tweak[31] = 1;
                secp256k1_ec_seckey_tweak_add(secp_ctx, privkey_i, tweak);
            }
            if (all_pass) {
                printf("  [PASS] 5.4 批量归一化 2048 点全部与公开 API 一致\n");
                pass_count++;
            }

            free(gej_arr);
            free(ge_arr);
        }
    }

    /* ------------------------------------------------------------------
     * 5.5  keygen_ge_to_pubkey_bytes 非压缩格式验证：
     *      验证 k=1 的非压缩公钥字节与公开 API serialize 一致
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey1[32] = {0};
        privkey1[31] = 1;

        /* 方法A：keygen 路径 */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t keygen_uncomp[65];
        keygen_privkey_to_gej(secp_ctx, privkey1, &gej);
        keygen_batch_normalize(&gej, &ge, 1);
        keygen_ge_to_pubkey_bytes(&ge, NULL, keygen_uncomp);

        /* 方法B：公开 API */
        secp256k1_pubkey pubkey;
        uint8_t api_uncomp[65];
        size_t len = 65;
        secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey1);
        secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len,
                                      &pubkey, SECP256K1_EC_UNCOMPRESSED);

        char api_hex[131];
        bytes_to_hex_helper(api_uncomp, 65, api_hex);
        check("5.5 keygen_ge_to_pubkey_bytes 非压缩格式与公开 API 一致（k=1）",
              api_hex, keygen_uncomp, 65);
    }

    /* ------------------------------------------------------------------
     * 5.6  keygen 路径的 hash160 与 privkey_to_hash160 一致性：
     *      验证 k=5 经过 keygen 路径得到的 hash160 与直接计算一致
     * ------------------------------------------------------------------ */
    {
        uint8_t privkey5[32] = {0};
        privkey5[31] = 5;

        /* 方法A：keygen 路径 */
        secp256k1_gej gej;
        secp256k1_ge  ge;
        uint8_t comp_bytes[33], uncomp_bytes[65];
        uint8_t hash160_keygen_comp[20], hash160_keygen_uncomp[20];
        keygen_privkey_to_gej(secp_ctx, privkey5, &gej);
        keygen_batch_normalize(&gej, &ge, 1);
        keygen_ge_to_pubkey_bytes(&ge, comp_bytes, uncomp_bytes);
        pubkey_bytes_to_hash160(comp_bytes,   33, hash160_keygen_comp);
        pubkey_bytes_to_hash160(uncomp_bytes, 65, hash160_keygen_uncomp);

        /* 方法B：privkey_to_hash160 */
        uint8_t hash160_direct_comp[20], hash160_direct_uncomp[20];
        privkey_to_hash160(privkey5, hash160_direct_comp, hash160_direct_uncomp);

        char direct_comp_hex[41], direct_uncomp_hex[41];
        bytes_to_hex_helper(hash160_direct_comp,   20, direct_comp_hex);
        bytes_to_hex_helper(hash160_direct_uncomp, 20, direct_uncomp_hex);

        check("5.6a keygen 路径压缩公钥 hash160 与 privkey_to_hash160 一致（k=5）",
              direct_comp_hex, hash160_keygen_comp, 20);
        check("5.6b keygen 路径非压缩公钥 hash160 与 privkey_to_hash160 一致（k=5）",
              direct_uncomp_hex, hash160_keygen_uncomp, 20);
    }

    /* ------------------------------------------------------------------
     * 5.7  多步点加法 + 批量归一化后的 hash160 一致性（10步）：
     *      从 k=3 连续点加 G 10 步，验证每步 hash160 与直接计算一致
     * ------------------------------------------------------------------ */
    {
        printf("  [多步点加法 + 批量归一化 hash160 验证，k=3..13]\n");

        const int STEPS = 10;
        secp256k1_gej gej_arr[10];
        secp256k1_ge  ge_arr[10];

        uint8_t privkey[32] = {0};
        privkey[31] = 3;
        secp256k1_gej cur;
        keygen_privkey_to_gej(secp_ctx, privkey, &cur);

        uint8_t tweak[32] = {0};
        tweak[31] = 1;

        /* 积累 10 个 Jacobian 点（k=4..13，即点加 G 后的结果） */
        for (int i = 0; i < STEPS; i++) {
            secp256k1_gej next;
            secp256k1_gej_add_ge(&next, &cur, &G_affine);
            cur = next;
            gej_arr[i] = cur;
        }

        /* 批量归一化 */
        keygen_batch_normalize(gej_arr, ge_arr, (size_t)STEPS);

        /* 逐步验证 hash160 */
        int all_pass = 1;
        uint8_t privkey_i[32] = {0};
        privkey_i[31] = 4;  /* 第一步对应 k=4 */
        for (int i = 0; i < STEPS; i++) {
            uint8_t comp_bytes[33];
            uint8_t hash160_keygen[20], hash160_direct[20];

            keygen_ge_to_pubkey_bytes(&ge_arr[i], comp_bytes, NULL);
            pubkey_bytes_to_hash160(comp_bytes, 33, hash160_keygen);
            privkey_to_hash160(privkey_i, hash160_direct, NULL);

            if (memcmp(hash160_keygen, hash160_direct, 20) != 0) {
                char kh[41], dh[41];
                bytes_to_hex_helper(hash160_keygen, 20, kh);
                bytes_to_hex_helper(hash160_direct, 20, dh);
                printf("  [FAIL] 5.7 步骤 %d (k=%d): keygen=%s direct=%s\n",
                       i, i + 4, kh, dh);
                all_pass = 0;
                fail_count++;
            }
            secp256k1_ec_seckey_tweak_add(secp_ctx, privkey_i, tweak);
        }
        if (all_pass) {
            printf("  [PASS] 5.7 多步点加法 + 批量归一化 hash160（10步全部一致）\n");
            pass_count++;
        }
    }
}

/* ------------------------------------------------------------------
 * test_search_key_privkey_pubkey
 *
 * 专项测试：模拟 search_key 中的私钥迭代与公钥推导流程，
 * 验证内部路径（scalar 累加 + gej_add_ge_var + batch_normalize_rzr）
 * 与标准 API（secp256k1_ec_pubkey_create + serialize）结果完全一致。
 *
 * 覆盖场景：
 *   6.1  单批次首尾两端（b=0 和 b=BATCH_SIZE-1）私钥→公钥一致性
 *   6.2  rzr 路径 batch_normalize_rzr 与 batch_normalize 结果一致
 *   6.3  命中重建逻辑：hit_scalar = base + b_idx * tweak 对应公钥
 *        与 ge_batch[b_idx] 一致（验证 b_idx=0/中间/末尾三个位置）
 *   6.4  完整批次（BATCH_SIZE=4096）每点公钥与标准 API 一致
 *   6.5  scalar 溢出边界：base 接近曲线阶 n，验证溢出检测正确
 *   6.6  非压缩公钥路径：ge_batch 的非压缩字节与标准 API 一致
 * ------------------------------------------------------------------ */
static void test_search_key_privkey_pubkey(void) {
    printf("\n=== search_key 私钥迭代与公钥推导一致性测试 ===\n");

    /* tweak_scalar = 1（与 search_key 中一致） */
    secp256k1_scalar tweak_scalar;
    secp256k1_scalar_set_int(&tweak_scalar, 1);

    /* 公开 API 用的 tweak 字节（值为 1） */
    uint8_t tweak_bytes[32] = {0};
    tweak_bytes[31] = 1;

    /* ------------------------------------------------------------------ */
    /* 6.1  单批次首尾两端（b=0 和 b=BATCH_SIZE-1）私钥→公钥一致性
     *
     *   模拟 search_key 外层循环：
     *     base_privkey_scalar = k（随机基准，此处取 k=7 便于验证）
     *     cur_privkey_scalar  = base_privkey_scalar
     *     gej_batch[0]        = keygen_privkey_to_gej(base)
     *     gej_batch[b]        = gej_batch[b-1] + G
     *
     *   验证：
     *     ge_batch[0]  对应私钥 base+0 的公钥
     *     ge_batch[N-1] 对应私钥 base+(N-1) 的公钥
     * ------------------------------------------------------------------ */
    {
        const int N = 16;  /* 小批次，快速验证首尾 */
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 7;  /* base = 7 */

        secp256k1_scalar base_scalar, cur_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);

        /* 构造 gej_batch 和 rzr_batch（模拟 search_key 内层循环） */
        secp256k1_gej gej_batch[16];
        secp256k1_ge  ge_batch[16];
        secp256k1_fe  rzr_batch[16];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);
        cur_scalar = base_scalar;

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak_scalar);
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }

        /* 批量归一化（rzr 路径） */
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        /* 验证 b=0：ge_batch[0] 对应私钥 base+0=7 */
        {
            uint8_t keygen_comp[33], api_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[0], keygen_comp, NULL);

            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, base_privkey);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            char api_hex[67];
            bytes_to_hex_helper(api_comp, 33, api_hex);
            check("6.1a 批次首端 b=0：ge_batch[0] 与标准 API（base=7）一致",
                  api_hex, keygen_comp, 33);
        }

        /* 验证 b=N-1：ge_batch[N-1] 对应私钥 base+(N-1)=7+15=22 */
        {
            uint8_t privkey_end[32] = {0};
            privkey_end[31] = 7 + (N - 1);  /* = 22 */

            uint8_t keygen_comp[33], api_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[N - 1], keygen_comp, NULL);

            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey_end);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);
            char api_hex[67];
            bytes_to_hex_helper(api_comp, 33, api_hex);
            check("6.1b 批次末端 b=N-1：ge_batch[N-1] 与标准 API（base+15=22）一致",
                  api_hex, keygen_comp, 33);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.2  rzr 路径 batch_normalize_rzr 与 batch_normalize 结果一致
     *
     *   对同一组 gej_batch，分别用两种归一化方式，
     *   验证每个点的压缩公钥字节完全相同。
     * ------------------------------------------------------------------ */
    {
        printf("  [rzr 路径 vs 标准路径 batch_normalize 一致性，N=32]\n");
        const int N = 32;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 100;  /* base = 100 */

        secp256k1_gej gej_batch[32];
        secp256k1_ge  ge_rzr[32], ge_std[32];
        secp256k1_fe  rzr_batch[32];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }

        /* 两种归一化 */
        keygen_batch_normalize_rzr(gej_batch, ge_rzr, rzr_batch, (size_t)N);
        keygen_batch_normalize(gej_batch, ge_std, (size_t)N);

        int all_pass = 1;
        for (int b = 0; b < N; b++) {
            uint8_t comp_rzr[33], comp_std[33];
            keygen_ge_to_pubkey_bytes(&ge_rzr[b], comp_rzr, NULL);
            keygen_ge_to_pubkey_bytes(&ge_std[b], comp_std, NULL);
            if (memcmp(comp_rzr, comp_std, 33) != 0) {
                char rzr_hex[67], std_hex[67];
                bytes_to_hex_helper(comp_rzr, 33, rzr_hex);
                bytes_to_hex_helper(comp_std, 33, std_hex);
                printf("  [FAIL] 6.2 b=%d: rzr=%s std=%s\n", b, rzr_hex, std_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.2 rzr 路径与标准路径 batch_normalize 结果完全一致（N=32）\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.3  命中重建逻辑验证
     *
     *   模拟 search_key 命中时的私钥重建：
     *     hit_scalar = base_privkey_scalar
     *     for i in range(b_idx): hit_scalar += tweak_scalar
     *     hit_privkey = scalar_get_b32(hit_scalar)
     *
     *   验证三个位置（b_idx=0, 中间, 末尾）：
     *     重建私钥对应的公钥 == ge_batch[b_idx] 的公钥字节
     * ------------------------------------------------------------------ */
    {
        const int N = 64;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 50;  /* base = 50 */

        secp256k1_scalar base_scalar;
        int overflow = 0;
        secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);

        secp256k1_gej gej_batch[64];
        secp256k1_ge  ge_batch[64];
        secp256k1_fe  rzr_batch[64];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        /* 测试三个命中位置 */
        int test_positions[] = {0, N / 2, N - 1};
        const char *pos_names[] = {"b_idx=0（首端）", "b_idx=N/2（中间）", "b_idx=N-1（末端）"};
        int all_pass = 1;

        for (int t = 0; t < 3; t++) {
            int b_idx = test_positions[t];

            /* 模拟命中重建：hit_scalar = base + b_idx * tweak */
            secp256k1_scalar hit_scalar = base_scalar;
            for (int i = 0; i < b_idx; i++) {
                secp256k1_scalar_add(&hit_scalar, &hit_scalar, &tweak_scalar);
            }
            uint8_t hit_privkey[32];
            secp256k1_scalar_get_b32(hit_privkey, &hit_scalar);

            /* 重建私钥对应的公钥（标准 API） */
            uint8_t api_comp[33];
            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, hit_privkey);
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                          SECP256K1_EC_COMPRESSED);

            /* ge_batch[b_idx] 的公钥字节 */
            uint8_t batch_comp[33];
            keygen_ge_to_pubkey_bytes(&ge_batch[b_idx], batch_comp, NULL);

            if (memcmp(api_comp, batch_comp, 33) != 0) {
                char api_hex[67], batch_hex[67];
                bytes_to_hex_helper(api_comp,   33, api_hex);
                bytes_to_hex_helper(batch_comp, 33, batch_hex);
                printf("  [FAIL] 6.3 命中重建 %s:\n"
                       "         重建私钥公钥: %s\n"
                       "         ge_batch公钥: %s\n",
                       pos_names[t], api_hex, batch_hex);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.3 命中重建逻辑：三个位置（首/中/末）私钥重建公钥与 ge_batch 一致\n");
            pass_count++;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.4  完整批次（BATCH_SIZE=4096）每点公钥与标准 API 一致
     *
     *   模拟 search_key 完整内层循环，验证所有 4096 个点的公钥正确性。
     *   采样验证策略：验证首点、末点、以及每 256 点采样一次（共 18 点）。
     * ------------------------------------------------------------------ */
    {
        printf("  [完整批次 BATCH_SIZE=4096 采样验证]\n");
        const int N = 4096;

        secp256k1_gej *gej_batch = (secp256k1_gej *)malloc(N * sizeof(secp256k1_gej));
        secp256k1_ge  *ge_batch  = (secp256k1_ge  *)malloc(N * sizeof(secp256k1_ge));
        secp256k1_fe  *rzr_batch = (secp256k1_fe  *)malloc(N * sizeof(secp256k1_fe));

        if (!gej_batch || !ge_batch || !rzr_batch) {
            printf("  [SKIP] 6.4 内存分配失败，跳过\n");
            free(gej_batch); free(ge_batch); free(rzr_batch);
        } else {
            /* base = 0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20 */
            uint8_t base_privkey[32];
            for (int i = 0; i < 32; i++) base_privkey[i] = (uint8_t)(i + 1);

            secp256k1_scalar base_scalar, cur_scalar;
            int overflow = 0;
            secp256k1_scalar_set_b32(&base_scalar, base_privkey, &overflow);
            cur_scalar = base_scalar;

            secp256k1_gej cur_gej;
            keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

            /* 构造完整批次（模拟 search_key 内层循环） */
            for (int b = 0; b < N; b++) {
                gej_batch[b] = cur_gej;
                if (b < N - 1) {
                    secp256k1_scalar_add(&cur_scalar, &cur_scalar, &tweak_scalar);
                    secp256k1_gej next_gej;
                    secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                    cur_gej = next_gej;
                }
            }

            /* rzr 路径批量归一化 */
            keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

            /* 采样验证：首点、末点、每 256 点一次 */
            int all_pass = 1;
            uint8_t cur_privkey[32];
            memcpy(cur_privkey, base_privkey, 32);

            for (int b = 0; b < N; b++) {
                /* 仅验证采样点 */
                int do_check = (b == 0) || (b == N - 1) || (b % 256 == 0);
                if (!do_check) {
                    /* 非采样点只推进私钥 */
                    secp256k1_ec_seckey_tweak_add(secp_ctx, cur_privkey, tweak_bytes);
                    continue;
                }

                uint8_t batch_comp[33], api_comp[33];
                keygen_ge_to_pubkey_bytes(&ge_batch[b], batch_comp, NULL);

                secp256k1_pubkey pubkey;
                secp256k1_ec_pubkey_create(secp_ctx, &pubkey, cur_privkey);
                size_t len = 33;
                secp256k1_ec_pubkey_serialize(secp_ctx, api_comp, &len, &pubkey,
                                              SECP256K1_EC_COMPRESSED);

                if (memcmp(batch_comp, api_comp, 33) != 0) {
                    char bh[67], ah[67];
                    bytes_to_hex_helper(batch_comp, 33, bh);
                    bytes_to_hex_helper(api_comp,   33, ah);
                    printf("  [FAIL] 6.4 b=%d: batch=%s api=%s\n", b, bh, ah);
                    all_pass = 0;
                    fail_count++;
                    break;
                }

                if (b < N - 1)
                    secp256k1_ec_seckey_tweak_add(secp_ctx, cur_privkey, tweak_bytes);
            }
            if (all_pass) {
                printf("  [PASS] 6.4 完整批次 4096 点采样验证（首/末/每256点）全部与标准 API 一致\n");
                pass_count++;
            }

            free(gej_batch); free(ge_batch); free(rzr_batch);
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.5  scalar 溢出边界检测
     *
     *   secp256k1 曲线阶 n（32字节大端）：
     *     FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
     *
     *   测试场景：
     *     a) base = n-1（最大有效私钥），验证 keygen_privkey_to_gej 成功
     *     b) base = n（等于阶），验证 scalar_set_b32 返回 overflow=1
     *     c) base = n-2，连续 +1 两步后 scalar 应归零（overflow），
     *        验证 search_key 中的 inner_overflow 检测逻辑
     * ------------------------------------------------------------------ */
    {
        /* secp256k1 曲线阶 n */
        static const uint8_t curve_n[32] = {
            0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
            0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
            0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
            0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
        };

        /* 6.5a：base = n-1，keygen_privkey_to_gej 应成功 */
        {
            uint8_t privkey_nm1[32];
            memcpy(privkey_nm1, curve_n, 32);
            /* n-1：最低字节 -1 */
            privkey_nm1[31] -= 1;

            secp256k1_gej gej;
            int ret = keygen_privkey_to_gej(secp_ctx, privkey_nm1, &gej);
            if (ret == 0) {
                printf("  [PASS] 6.5a base=n-1：keygen_privkey_to_gej 成功（有效私钥）\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5a base=n-1：keygen_privkey_to_gej 意外失败\n");
                fail_count++;
            }
        }

        /* 6.5b：base = n，scalar_set_b32 应返回 overflow=1 */
        {
            secp256k1_scalar s;
            int overflow = 0;
            secp256k1_scalar_set_b32(&s, curve_n, &overflow);
            if (overflow) {
                printf("  [PASS] 6.5b base=n：scalar_set_b32 正确检测 overflow\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5b base=n：scalar_set_b32 未检测到 overflow\n");
                fail_count++;
            }
        }

        /* 6.5c：base = n-2，连续 +1 两步后 scalar 应归零
         *   第1步：n-2 + 1 = n-1（非零，正常）
         *   第2步：n-1 + 1 = n ≡ 0 (mod n)，scalar_is_zero 应为真
         *   这对应 search_key 中 inner_overflow 检测的触发条件
         */
        {
            uint8_t privkey_nm2[32];
            memcpy(privkey_nm2, curve_n, 32);
            privkey_nm2[31] -= 2;  /* n-2 */

            secp256k1_scalar s;
            int overflow = 0;
            secp256k1_scalar_set_b32(&s, privkey_nm2, &overflow);

            /* 第1步：+1，应非零 */
            secp256k1_scalar_add(&s, &s, &tweak_scalar);
            int step1_zero = secp256k1_scalar_is_zero(&s);

            /* 第2步：+1，应归零 */
            secp256k1_scalar_add(&s, &s, &tweak_scalar);
            int step2_zero = secp256k1_scalar_is_zero(&s);

            if (!overflow && !step1_zero && step2_zero) {
                printf("  [PASS] 6.5c base=n-2：两步后 scalar 归零，inner_overflow 检测逻辑正确\n");
                pass_count++;
            } else {
                printf("  [FAIL] 6.5c base=n-2：overflow=%d step1_zero=%d step2_zero=%d（期望 0,0,1）\n",
                       overflow, step1_zero, step2_zero);
                fail_count++;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* 6.6  非压缩公钥路径：ge_batch 的非压缩字节与标准 API 一致
     *
     *   验证 keygen_ge_to_pubkey_bytes 的非压缩输出（65字节）
     *   与 secp256k1_ec_pubkey_serialize(UNCOMPRESSED) 完全一致。
     *   测试 b=0, b=7, b=15 三个位置。
     * ------------------------------------------------------------------ */
    {
        const int N = 16;
        uint8_t base_privkey[32] = {0};
        base_privkey[31] = 200;  /* base = 200 */

        secp256k1_gej gej_batch[16];
        secp256k1_ge  ge_batch[16];
        secp256k1_fe  rzr_batch[16];

        secp256k1_gej cur_gej;
        keygen_privkey_to_gej(secp_ctx, base_privkey, &cur_gej);

        for (int b = 0; b < N; b++) {
            gej_batch[b] = cur_gej;
            if (b < N - 1) {
                secp256k1_gej next_gej;
                secp256k1_gej_add_ge_var(&next_gej, &cur_gej, &G_affine, &rzr_batch[b]);
                cur_gej = next_gej;
            }
        }
        keygen_batch_normalize_rzr(gej_batch, ge_batch, rzr_batch, (size_t)N);

        int test_positions[] = {0, 7, 15};
        const char *pos_names[] = {"b=0", "b=7", "b=15"};
        int all_pass = 1;

        for (int t = 0; t < 3; t++) {
            int b = test_positions[t];

            /* keygen 路径非压缩公钥 */
            uint8_t keygen_uncomp[65];
            keygen_ge_to_pubkey_bytes(&ge_batch[b], NULL, keygen_uncomp);

            /* 标准 API 非压缩公钥（私钥 = base + b） */
            uint8_t privkey_b[32] = {0};
            privkey_b[31] = (uint8_t)(200 + b);
            secp256k1_pubkey pubkey;
            secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey_b);
            uint8_t api_uncomp[65];
            size_t len = 65;
            secp256k1_ec_pubkey_serialize(secp_ctx, api_uncomp, &len, &pubkey,
                                          SECP256K1_EC_UNCOMPRESSED);

            if (memcmp(keygen_uncomp, api_uncomp, 65) != 0) {
                char kh[131], ah[131];
                bytes_to_hex_helper(keygen_uncomp, 65, kh);
                bytes_to_hex_helper(api_uncomp,    65, ah);
                printf("  [FAIL] 6.6 非压缩公钥 %s:\n"
                       "         keygen: %s\n"
                       "         api:    %s\n",
                       pos_names[t], kh, ah);
                all_pass = 0;
                fail_count++;
            }
        }
        if (all_pass) {
            printf("  [PASS] 6.6 非压缩公钥路径（b=0/7/15）与标准 API 完全一致\n");
            pass_count++;
        }
    }
}
#endif /* USE_PUBKEY_API_ONLY */

/* ===================== main ===================== */

int main(void) {
    printf("========================================\n");
    printf("  keysearch 正确性验证程序\n");
    printf("========================================\n");

    secp_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!secp_ctx) {
        fprintf(stderr, "错误：初始化 secp256k1 失败\n");
        return 1;
    }

    test_sha256();
    test_ripemd160();
    test_hash160();
    test_incremental_pubkey();
#ifdef __AVX2__
    test_avx2_compress();
    test_hash160_8way();
#endif
#ifdef __AVX512F__
    if (__builtin_cpu_supports("avx512f")) {
        test_avx512_compress();
        test_hash160_16way();
    }
#endif
    test_ht_openaddr();
#ifdef __AVX2__
    test_ht_contains_8way_func();
#endif
#ifdef __AVX512F__
    if (__builtin_cpu_supports("avx512f")) {
        test_ht_contains_16way_func();
    }
#endif    test_specialized_interfaces();
#ifndef USE_PUBKEY_API_ONLY
    /* 初始化全局生成元 G（供 test_keygen_internal / test_search_key_privkey_pubkey 使用） */
    if (keygen_init_generator(secp_ctx, &G_affine) != 0) {
        fprintf(stderr, "错误：keygen_init_generator 失败，跳过内部接口测试\n");
    } else {
        test_keygen_internal();
        test_search_key_privkey_pubkey();
    }
#endif

    secp256k1_context_destroy(secp_ctx);

    printf("\n========================================\n");
    printf("  通过: %d 项  失败: %d 项\n", pass_count, fail_count);
    printf("========================================\n");

    return (fail_count > 0) ? 1 : 0;
}

