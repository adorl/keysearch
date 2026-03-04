/*
 * 依赖库：
 *   - libsecp256k1  (椭圆曲线公钥计算)
 *   - pthread       (多线程)
 *   SHA256 与 RIPEMD160 均已内嵌纯 C 实现，无需 OpenSSL
 * 用法：
 *   ./keysearch <地址文件> [线程数]
 */
#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#include <secp256k1.h>

#include "sha256.h"
#include "ripemd160.h"
#include "hash_utils.h"
#include "rand_key.h"

/* ===================== 常量定义 ===================== */
#define MAX_ATTEMPTS        (1ULL << 63)    /* 每个线程最大尝试次数 */
#define PROGRESS_INTERVAL   (1000000)       /* 进度打印间隔 */
#define MAX_ADDRESSES       (400000)        /* 最多支持的目标地址数量 */
#define HASH_TABLE_SIZE     (65536 * 4)     /* 哈希表固定槽位数 */
#define ADDRESS_LEN         (35)            /* 比特币地址最大长度 */
#define BATCH_SIZE          (32 * 1024)     /* 增量推导批次大小，每批后重置随机基准私钥 */

/* ===================== 全局共享数据 ===================== */

/* 哈希表节点（用于O(1)查找） */
struct hash_node
{
    uint8_t hash160[20];        /* 20字节RIPEMD160哈希值 */
    struct hash_node *next;
};

static struct hash_node *hash_table[HASH_TABLE_SIZE]; /* 地址哈希表 */
static int address_count = 0;                /* 已加载地址数量 */

/* 跨线程找到标志 */
static volatile int found_flag = 0;

/* secp256k1上下文（只读，多线程安全） */
secp256k1_context *secp_ctx = NULL;

struct thread_args
{
    int thread_id;
};

/* 基于20字节二进制数据的哈希函数 */
static uint32_t hash160_hash(const uint8_t *h160)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < 20; i++) {
        h ^= h160[i];
        h *= 16777619u;
    }
    return h & (HASH_TABLE_SIZE - 1);
}

/* 向哈希表插入hash160 */
static void ht_insert(const uint8_t *h160)
{
    uint32_t idx = hash160_hash(h160);
    struct hash_node *node = (struct hash_node *)malloc(sizeof(struct hash_node));

    memcpy(node->hash160, h160, 20);
    node->next = hash_table[idx];
    hash_table[idx] = node;
}

/* 在哈希表中查找hash160，O(1)平均 */
static int ht_contains(const uint8_t *h160)
{
    uint32_t idx = hash160_hash(h160);
    struct hash_node *node = hash_table[idx];
    while (node) {
        if (memcmp(node->hash160, h160, 20) == 0)
            return 1;
        node = node->next;
    }
    return 0;
}

/* ===================== 线程工作函数 ===================== */
static void *search_key(void *arg)
{
    struct thread_args *args = (struct thread_args *)arg;
    int thread_id = args->thread_id;

    uint8_t privkey[32];
    uint8_t pubkey_compressed[33];      /* 压缩公钥序列化缓冲区 */
    uint8_t pubkey_uncompressed[65];    /* 非压缩公钥序列化缓冲区 */
    uint8_t hash160_compressed[20];
    uint8_t hash160_uncompressed[20];
    secp256k1_pubkey pubkey;            /* 当前公钥（内部表示） */
    uint8_t tweak[32][32];              /* 标量加法tweak = 1 */
    uint64_t count = 0;
    rand_key_context rand_ctx;

    /* 用于每步私钥+1和公钥点加G */
    memset(tweak, 0, sizeof(tweak));
    for (int i = 0; i < 32; i++)
        tweak[i][31 - i] = 1;

    /* 初始化真随机上下文（每线程独立fd，无锁竞争） */
    if (rand_ctx_init(&rand_ctx) != 0) {
        fprintf(stderr, "[线程-%d] 打开/dev/urandom失败\n", thread_id);
        return NULL;
    }

    /* 性能监控：记录上一次打印时间 */
    struct timespec ts_last, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_last);
    uint64_t count_last = 0;

    while (count < MAX_ATTEMPTS) {
        /* 生成随机私钥 */
        if (gen_random_key(privkey, &rand_ctx) != 0) {
            fprintf(stderr, "[线程-%d] 读取随机数失败\n", thread_id);
            break;
        }

        if (!secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey))
            continue;

        /* 内层循环：增量推导BATCH_SIZE次 */
        for (int batch = 0; batch < BATCH_SIZE && count < MAX_ATTEMPTS; batch++) {
            size_t len_comp = 33;
            size_t len_uncomp = 65;
            int tweak_index = batch / 1024;

            count++;

            /* 序列化公钥 */
            secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_compressed, &len_comp,
                                          &pubkey, SECP256K1_EC_COMPRESSED);
            secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_uncompressed, &len_uncomp,
                                          &pubkey, SECP256K1_EC_UNCOMPRESSED);

            /* 直接从公钥字节计算hash160 */
            pubkey_bytes_to_hash160(pubkey_compressed, 33, hash160_compressed);
            pubkey_bytes_to_hash160(pubkey_uncompressed, 65, hash160_uncompressed);

            /* 同时比对压缩地址和非压缩地址（字节层面直接比对） */
            if (ht_contains(hash160_compressed) || ht_contains(hash160_uncompressed)) {
                found_flag = 1;
                /* 仅在命中时才做格式转换 */
                char privkey_hex[65];
                char address_compressed[ADDRESS_LEN + 1];
                char address_uncompressed[ADDRESS_LEN + 1];
                bytes_to_hex(privkey, 32, privkey_hex);
                fprintf(stdout, "\n[线程-%d] 找到匹配！总尝试次数: %lu\n", thread_id, count);
                fprintf(stdout, "私钥(hex): %s\n", privkey_hex);
                if (privkey_to_address(privkey, address_compressed, address_uncompressed) == 0) {
                    fprintf(stdout, "压缩地址: %s\n", address_compressed);
                    fprintf(stdout, "非压缩地址: %s\n", address_uncompressed);
                }
                fflush(stdout);
            }

            /* 性能监控：每 PROGRESS_INTERVAL 次输出 keys/s */
            if (count % PROGRESS_INTERVAL == 0) {
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                double elapsed = (ts_now.tv_sec - ts_last.tv_sec) + (ts_now.tv_nsec - ts_last.tv_nsec) * 1e-9;
                double kps = (elapsed > 0) ? (double)(count - count_last) / elapsed : 0.0;
                fprintf(stdout, "[线程-%d] 已尝试: %lu  速度: %.0f keys/s\n", thread_id, count, kps);
                fflush(stdout);
                ts_last = ts_now;
                count_last = count;
            }

            /* 增量推导：私钥+1，公钥点加G */
            if (!secp256k1_ec_seckey_tweak_add(secp_ctx, privkey, tweak[tweak_index])) {
                fprintf(stderr, "警告：私钥推导失败，batch=%d！\n", batch);
                break;
            }
            if (!secp256k1_ec_pubkey_tweak_add(secp_ctx, &pubkey, tweak[tweak_index])) {
                fprintf(stderr, "警告：公钥推导失败，batch=%d！\n", batch);
                break;
            }
        }
    }

    if (count >= MAX_ATTEMPTS) {
        fprintf(stdout, "[线程-%d] 已达到最大尝试次数，退出。\n", thread_id);
    }

    fflush(stdout);
    return NULL;
}

/* ===================== 加载地址文件 ===================== */
static int load_target_addresses(const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "错误：文件%s不存在！\n", filename);
        return -1;
    }

    char line[ADDRESS_LEN + 2];
    int count = 0;
    int skip_count = 0;
    while (fgets(line, sizeof(line), f)) {
        /* 去除换行符 */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0)
            continue;
        if (count >= MAX_ADDRESSES) {
            fprintf(stderr, "警告：地址数量超过上限%d，忽略多余地址\n", MAX_ADDRESSES);
            break;
        }

        /* 将地址解码为20字节hash160后存入哈希表 */
        uint8_t h160[20];
        int ret = base58check_decode(line, h160);
        if (ret != 0) {
            fprintf(stderr, "警告：地址解码失败（ret=%d），跳过：%s\n", ret, line);
            skip_count++;
            continue;
        }
        ht_insert(h160);
        count++;
    }

    fclose(f);

    if (count == 0) {
        fprintf(stderr, "错误：文件%s中没有有效地址！\n", filename);
        return -1;
    }
    address_count = count;
    fprintf(stdout, "成功加载%d个地址（跳过%d个无效地址）\n", count, skip_count);

    fflush(stdout);
    return 0;
}

/* ===================== 主函数 ===================== */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("用法: ./test <地址文件> [线程数]\n");
        printf("  地址文件: 每行一个目标比特币地址\n");
        printf("  线程数:   可选，默认为 4\n");
        return 1;
    }

    const char *address_file = argv[1];
    int thread_count = (argc >= 3) ? atoi(argv[2]) : 4;
    if (thread_count <= 0)
        thread_count = 4;

    /* 初始化哈希表 */
    memset(hash_table, 0, sizeof(hash_table));

    /* 加载目标地址 */
    if (load_target_addresses(address_file) != 0)
        return 1;

    fprintf(stdout, "已加载%d个目标地址，启动%d个线程开始查找...\n",
           address_count, thread_count);

    /* 初始化secp256k1上下文（SIGN用于创建公钥） */
    secp_ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!secp_ctx) {
        fprintf(stderr, "错误：初始化secp256k1失败\n");
        return 1;
    }

    /* 启动线程 */
    pthread_t *threads = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    struct thread_args *args = (struct thread_args *)malloc(thread_count * sizeof(struct thread_args));

    for (int i = 0; i < thread_count; i++) {
        args[i].thread_id = i + 1;
        pthread_create(&threads[i], NULL, search_key, &args[i]);
    }

    /* 等待所有线程结束 */
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    if (!found_flag) {
        fprintf(stdout, "所有线程均已达到最大尝试次数，未找到匹配地址。\n");
    }

    /* 清理资源 */
    secp256k1_context_destroy(secp_ctx);
    free(threads);
    free(args);

    fflush(stdout);
    return 0;
}

