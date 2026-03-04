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
#ifndef USE_PUBKEY_API_ONLY
    /* 初始化全局生成元 G（供 test_keygen_internal 使用） */
    if (keygen_init_generator(secp_ctx, &G_affine) != 0) {
        fprintf(stderr, "错误：keygen_init_generator 失败，跳过内部接口测试\n");
    } else {
        test_keygen_internal();
    }
#endif

    secp256k1_context_destroy(secp_ctx);

    printf("\n========================================\n");
    printf("  通过: %d 项  失败: %d 项\n", pass_count, fail_count);
    printf("========================================\n");

    return (fail_count > 0) ? 1 : 0;
}

