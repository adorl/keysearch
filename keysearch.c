/*
 * test.c - 比特币私钥暴力搜索工具（C语言多线程版本）
 *
 * 依赖库：
 *   - libsecp256k1  (椭圆曲线公钥计算)
 *   - pthread       (多线程)
 *   SHA256 与 RIPEMD160 均已内嵌纯 C 实现，无需 OpenSSL
 *
 * 编译：
 *   gcc -O2 -std=c99 -o test test.c -lsecp256k1 -lpthread
 *
 * 用法：
 *   ./test <地址文件> [线程数]
 */
#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <secp256k1.h>

/* ===================== 内嵌 SHA256 实现 ===================== */

typedef struct {
    uint32_t state[8];
    uint32_t count[2];
    uint8_t  buf[64];
} sha256_ctx;

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

#define SHA256_CH(x,y,z)   (((x)&(y))^(~(x)&(z)))
#define SHA256_MAJ(x,y,z)  (((x)&(y))^((x)&(z))^((y)&(z)))
#define SHA256_EP0(x)  (ROR32(x,2)  ^ ROR32(x,13) ^ ROR32(x,22))
#define SHA256_EP1(x)  (ROR32(x,6)  ^ ROR32(x,11) ^ ROR32(x,25))
#define SHA256_SIG0(x) (ROR32(x,7)  ^ ROR32(x,18) ^ ((x)>>3))
#define SHA256_SIG1(x) (ROR32(x,17) ^ ROR32(x,19) ^ ((x)>>10))
#define ROR32(x,n)     (((x)>>(n))|((x)<<(32-(n))))

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

static void sha256_init(sha256_ctx *ctx)
{
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
    ctx->count[0] = ctx->count[1] = 0;
}

static void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len)
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

static void sha256_final(sha256_ctx *ctx, uint8_t *digest)
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

static void sha256(const uint8_t *data, size_t len, uint8_t *digest)
{
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}

/* ===================== 内嵌 RIPEMD160 实现 ===================== */
/* 参考 Bitcoin Core / RFC 2286，无需 OpenSSL RIPEMD160 接口 */

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

static const uint32_t RMD_KL[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const uint32_t RMD_KR[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
static const int RMD_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
static const int RMD_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
static const int RMD_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
static const int RMD_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

static void ripemd160_compress(uint32_t *state, const uint8_t *block)
{
    uint32_t w[16];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)block[i*4] | ((uint32_t)block[i*4+1]<<8)
             | ((uint32_t)block[i*4+2]<<16) | ((uint32_t)block[i*4+3]<<24);

    uint32_t al=state[0],bl=state[1],cl=state[2],dl=state[3],el=state[4];
    uint32_t ar=state[0],br=state[1],cr=state[2],dr=state[3],er=state[4];

    for (int i = 0; i < 80; i++) {
        int r = i / 16;
        uint32_t fl, fr, t;
        switch(r) {
            case 0: fl=RMD_F(bl,cl,dl); fr=RMD_J(br,cr,dr); break;
            case 1: fl=RMD_G(bl,cl,dl); fr=RMD_I(br,cr,dr); break;
            case 2: fl=RMD_H(bl,cl,dl); fr=RMD_H(br,cr,dr); break;
            case 3: fl=RMD_I(bl,cl,dl); fr=RMD_G(br,cr,dr); break;
            default:fl=RMD_J(bl,cl,dl); fr=RMD_F(br,cr,dr); break;
        }
        t = ROL32(al + fl + w[RMD_RL[i]] + RMD_KL[r], RMD_SL[i]) + el;
        al=el; el=dl; dl=ROL32(cl,10); cl=bl; bl=t;
        t = ROL32(ar + fr + w[RMD_RR[i]] + RMD_KR[r], RMD_SR[i]) + er;
        ar=er; er=dr; dr=ROL32(cr,10); cr=br; br=t;
    }
    uint32_t t = state[1] + cl + dr;
    state[1] = state[2] + dl + er;
    state[2] = state[3] + el + ar;
    state[3] = state[4] + al + br;
    state[4] = state[0] + bl + cr;
    state[0] = t;
}

static void ripemd160_init(ripemd160_ctx *ctx)
{
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
    ctx->state[4] = 0xC3D2E1F0;
    ctx->count[0] = ctx->count[1] = 0;
}

static void ripemd160_update(ripemd160_ctx *ctx, const uint8_t *data, size_t len)
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

static void ripemd160_final(ripemd160_ctx *ctx, uint8_t *digest)
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

static void ripemd160(const uint8_t *data, size_t len, uint8_t *digest)
{
    ripemd160_ctx ctx;
    ripemd160_init(&ctx);
    ripemd160_update(&ctx, data, len);
    ripemd160_final(&ctx, digest);
}

/* ===================== 常量定义 ===================== */
#define MAX_ATTEMPTS        (1ULL << 48)    /* 每个线程最大尝试次数 2^32 */
#define PROGRESS_INTERVAL   (1000000)       /* 进度打印间隔 */
#define MAX_ADDRESSES       (400000)        /* 最多支持的目标地址数量 */
#define HASH_TABLE_SIZE     (65536 * 4)     /* 哈希表固定槽位数 */
#define ADDRESS_LEN         (35)            /* 比特币地址最大长度 */

/* ===================== Base58 字符表 ===================== */
static const char BASE58_CHARS[] =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* ===================== 全局共享数据 ===================== */

/* 哈希表节点（用于 O(1) 地址查找） */
struct hash_node
{
    char address[ADDRESS_LEN + 1];
    struct hash_node *next;
};

static struct hash_node *hash_table[HASH_TABLE_SIZE]; /* 地址哈希表 */
static int address_count = 0;                /* 已加载地址数量 */

/* 跨线程找到标志 */
static volatile int found_flag = 0;

/* secp256k1 上下文（只读，多线程安全） */
static secp256k1_context *secp_ctx = NULL;

/* ===================== 线程参数 ===================== */
struct thread_args
{
    int thread_id;
};

/* ===================== 工具函数 ===================== */

/* 简单字符串哈希（djb2） */
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

/* 在哈希表中查找地址，O(1) 平均 */
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

/* 每线程随机缓冲区大小：一次读取8KB，可供8192-32次私钥生成使用 */
#define RAND_BUF_SIZE   (8192)

struct rand_key_context {
    uint8_t  buf[RAND_BUF_SIZE];
    size_t   pos;
    int      fd;
};

/* 初始化随机上下文，打开/dev/urandom */
static int rand_ctx_init(struct rand_key_context *ctx)
{
    ctx->fd = open("/dev/urandom", O_RDONLY);
    if (ctx->fd < 0)
        return -1;
    ctx->pos = RAND_BUF_SIZE; /* 触发首次填充 */
    return 0;
}

/* 填充缓冲区 */
static int rand_ctx_refill(struct rand_key_context *ctx)
{
    ssize_t n = read(ctx->fd, ctx->buf, RAND_BUF_SIZE);
    if (n != RAND_BUF_SIZE)
        return -1;
    ctx->pos = 0;
    return 0;
}

/* 生成32字节真随机私钥（批量缓冲，减少syscall次数） */
static int gen_random_key(uint8_t *key32, struct rand_key_context *ctx)
{
    if (ctx->pos + 32 > RAND_BUF_SIZE) {
        if (rand_ctx_refill(ctx) != 0)
            return -1;
    }
    memcpy(key32, ctx->buf + ctx->pos, 32);
    ctx->pos += 1;
    return 0;
}

/* SHA256 两次哈希 */
static void sha256d(const uint8_t *data, size_t len, uint8_t *out)
{
    uint8_t tmp[32];
    sha256(data, len, tmp);
    sha256(tmp, 32, out);
}

/* Base58Check 编码 */
static void base58check_encode(const uint8_t *payload, size_t payload_len, char *out)
{
    /* 计算校验和：sha256d 前 4 字节 */
    uint8_t checksum[32];
    sha256d(payload, payload_len, checksum);

    /* 拼接 payload + checksum[0..3] */
    uint8_t buf[payload_len + 4];
    memcpy(buf, payload, payload_len);
    memcpy(buf + payload_len, checksum, 4);
    size_t total = payload_len + 4;

    /* 统计前导零字节 */
    int leading_zeros = 0;
    for (size_t i = 0; i < total && buf[i] == 0; i++)
        leading_zeros++;

    /* 大数转 Base58 */
    uint8_t digits[128] = {0};
    int digits_len = 0;

    for (size_t i = 0; i < total; i++) {
        int carry = buf[i];
        for (int j = 0; j < digits_len; j++) {
            carry += 256 * digits[j];
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry) {
            digits[digits_len++] = carry % 58;
            carry /= 58;
        }
    }

    /* 构建输出字符串（逆序） */
    int pos = 0;
    for (int i = 0; i < leading_zeros; i++)
        out[pos++] = '1';
    for (int i = digits_len - 1; i >= 0; i--)
        out[pos++] = BASE58_CHARS[digits[i]];
    out[pos] = '\0';
}

/*
 * 从 32 字节私钥同时计算压缩与非压缩两种比特币地址（P2PKH）
 * 流程：私钥 -> 公钥(一次椭圆曲线运算) -> 分别序列化为压缩(33B)/非压缩(65B)
 *       -> 各自 SHA256 -> RIPEMD160 -> 加版本号 0x00 -> Base58Check
 * compressed_out   : 压缩地址输出缓冲区（至少 ADDRESS_LEN+1 字节）
 * uncompressed_out : 非压缩地址输出缓冲区（至少 ADDRESS_LEN+1 字节）
 */
static int privkey_to_address(const uint8_t *privkey,
                              char *compressed_out,
                              char *uncompressed_out)
{
    uint8_t sha256_result[32];
    uint8_t hash160[20];
    uint8_t versioned[21];

    /* 1. 一次椭圆曲线运算，得到公钥对象 */
    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(secp_ctx, &pubkey, privkey))
        return -1;

    /* ---- 压缩地址 ---- */
    uint8_t pubkey_compressed[33];
    size_t pubkey_len1 = 33;
    secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_compressed, &pubkey_len1,
                                    &pubkey, SECP256K1_EC_COMPRESSED);

    /* 2. SHA256(公钥) */
    sha256(pubkey_compressed, 33, sha256_result);
    /* 3. RIPEMD160(SHA256结果) => 20字节 Hash160 */
    ripemd160(sha256_result, 32, hash160);
    /* 4. 加版本号 0x00（主网 P2PKH） */
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160, 20);
    base58check_encode(versioned, 21, compressed_out);

    /* ---- 非压缩地址 ---- */
    uint8_t pubkey_uncompressed[65];
    size_t pubkey_len2 = 65;
    secp256k1_ec_pubkey_serialize(secp_ctx, pubkey_uncompressed, &pubkey_len2,
                                    &pubkey, SECP256K1_EC_UNCOMPRESSED);
    sha256(pubkey_uncompressed, 65, sha256_result);
    ripemd160(sha256_result, 32, hash160);
    versioned[0] = 0x00;
    memcpy(versioned + 1, hash160, 20);
    base58check_encode(versioned, 21, uncompressed_out);

    return 0;
}

/* 私钥字节数组转十六进制字符串 */
static void bytes_to_hex(const uint8_t *bytes, int len, char *hex_out)
{
    for (int i = 0; i < len; i++) {
        sprintf(hex_out + i * 2, "%02x", bytes[i]);
    }
    hex_out[len * 2] = '\0';
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
    struct rand_key_context rand_ctx;

    /* 初始化真随机上下文（每线程独立 fd，无锁竞争） */
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

    /* 初始化 secp256k1 上下文（SIGN 用于创建公钥） */
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
