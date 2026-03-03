/*
 * test.c - 比特币私钥暴力搜索工具（C语言多线程版本）
 *
 * 依赖库：
 *   - libsecp256k1  (椭圆曲线公钥计算)
 *   - pthread       (多线程)
 *   SHA256 与 RIPEMD160 均已内嵌纯 C 实现，无需 OpenSSL
 *
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
#define MAX_ATTEMPTS        (1ULL << 48)    /* 每个线程最大尝试次数 */
#define PROGRESS_INTERVAL   (1000000)       /* 进度打印间隔 */
#define MAX_ADDRESSES       (400000)        /* 最多支持的目标地址数量 */
#define HASH_TABLE_SIZE     (65536 * 4)     /* 哈希表固定槽位数 */
#define ADDRESS_LEN         (35)            /* 比特币地址最大长度 */

/* ===================== 全局共享数据 ===================== */

/* 哈希表节点（用于O(1)地址查找） */
struct hash_node
{
    char address[ADDRESS_LEN + 1];
    struct hash_node *next;
};

static struct hash_node *hash_table[HASH_TABLE_SIZE]; /* 地址哈希表 */
static int address_count = 0;                /* 已加载地址数量 */

/* 跨线程找到标志 */
static volatile int found_flag = 0;

/* secp256k1上下文（只读，多线程安全） */
secp256k1_context *secp_ctx = NULL;

/* ===================== 线程参数 ===================== */
struct thread_args
{
    int thread_id;
};

/* ===================== 工具函数 ===================== */

/* 简单字符串哈希 */
static uint32_t str_hash(const char *s)
{
    uint32_t h = 5381;
    while (*s) {
        h = ((h << 5) + h) ^ (uint8_t)*s++;
    }
    return h & (HASH_TABLE_SIZE - 1);
}

/* 向哈希表插入地址 */
static void ht_insert(const char *addr)
{
    uint32_t idx = str_hash(addr);
    struct hash_node *node = (struct hash_node *)malloc(sizeof(struct hash_node));

    strncpy(node->address, addr, ADDRESS_LEN);
    node->address[ADDRESS_LEN] = '\0';
    node->next = hash_table[idx];
    hash_table[idx] = node;
}

/* 在哈希表中查找地址，O(1)平均 */
static int ht_contains(const char *addr)
{
    uint32_t idx = str_hash(addr);
    struct hash_node *node = hash_table[idx];
    while (node) {
        if (strcmp(node->address, addr) == 0)
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
    char address_compressed[ADDRESS_LEN + 1];
    char address_uncompressed[ADDRESS_LEN + 1];
    char privkey_hex[65];
    uint64_t count = 0;
    rand_key_context rand_ctx;

    /* 初始化真随机上下文（每线程独立fd，无锁竞争） */
    if (rand_ctx_init(&rand_ctx) != 0) {
        fprintf(stderr, "[线程-%d] 打开/dev/urandom失败\n", thread_id);
        return NULL;
    }

    while (count < MAX_ATTEMPTS) {
        /* 生成随机私钥 */
        if (gen_random_key(privkey, &rand_ctx) != 0) {
            fprintf(stderr, "[线程-%d] 读取随机数失败\n", thread_id);
            break;
        }
        count++;

        /* 同时计算压缩地址和非压缩地址 */
        if (privkey_to_address(privkey, address_compressed, address_uncompressed) != 0)
            continue;

        /* 打印进度 */
        if (count % PROGRESS_INTERVAL == 0) {
            fprintf(stdout, "[线程-%d] 已尝试次数: %llu\n", thread_id, (unsigned long long)count);
            fflush(stdout);
        }

        /* 同时比对压缩地址和非压缩地址 */
        if (ht_contains(address_compressed) || ht_contains(address_uncompressed)) {
            found_flag = 1;
            bytes_to_hex(privkey, 32, privkey_hex);
            fprintf(stdout, "\n[线程-%d] 找到匹配！总尝试次数: %llu\n",
                    thread_id, (unsigned long long)count);
            fprintf(stdout, "私钥(hex): %s\n", privkey_hex);
            fprintf(stdout, "压缩地址: %s\n", address_compressed);
            fprintf(stdout, "非压缩地址: %s\n", address_uncompressed);
            fflush(stdout);
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
        ht_insert(line);
        count++;
    }
    fclose(f);

    if (count == 0) {
        fprintf(stderr, "错误：文件%s中没有有效地址！\n", filename);
        return -1;
    }
    address_count = count;
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

