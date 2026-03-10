/*
 * gpu/gpu_hash160.cu
 *
 * GPU-side SHA256 + RIPEMD160 batch computation (Hash160):
 *   - Performs SHA256 on each public key (compressed 33 bytes + uncompressed 65 bytes)
 *   - Performs RIPEMD160 on SHA256 result to get Hash160 (20 bytes)
 *   - Public key data comes directly from gpu_secp256k1.cu device buffers, not transferred to CPU
 *
 * Requirements: 4.1, 4.2, 4.3, 4.4
 */

#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gpu_interface.h"
#include "../keylog.h"


__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __constant__ uint32_t SHA256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x,y,z)    (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z)   (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)       (ROTR32(x,2)  ^ ROTR32(x,13) ^ ROTR32(x,22))
#define EP1(x)       (ROTR32(x,6)  ^ ROTR32(x,11) ^ ROTR32(x,25))
#define SIG0(x)      (ROTR32(x,7)  ^ ROTR32(x,18) ^ ((x) >> 3))
#define SIG1(x)      (ROTR32(x,17) ^ ROTR32(x,19) ^ ((x) >> 10))

/* Read big-endian uint32 from byte array */
__device__ __forceinline__ uint32_t be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
}

/* Write big-endian uint32 */
__device__ __forceinline__ void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8); p[3] = (uint8_t)(v);
}

/* Write little-endian uint32 */
__device__ __forceinline__ void put_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);       p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/*
 * SHA256 single-block compression (64-byte input, updates state[8])
 * block : 64-byte input block (padding already applied)
 * state : 8 uint32 state values (input/output)
 */
__device__ void sha256_compress(const uint8_t *block, uint32_t state[8])
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = be32(block + i * 4);
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e,f,g) + SHA256_K[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/*
 * SHA256 computation (single block, input pre-padded to 64-byte padded block)
 * block  : 64-byte padded block
 * digest : 32-byte output
 */
__device__ void sha256_single_block(const uint8_t *block, uint8_t *digest)
{
    uint32_t state[8];
    for (int i = 0; i < 8; i++)
        state[i] = SHA256_INIT[i];
    sha256_compress(block, state);
    for (int i = 0; i < 8; i++)
        put_be32(digest + i * 4, state[i]);
}

/*
 * SHA256 computation (two blocks, for 65-byte uncompressed public key)
 * block1 : first 64-byte block (first 64 bytes of pubkey)
 * block2 : second 64-byte block (65th byte of pubkey + padding)
 * digest : 32-byte output
 */
__device__ void sha256_two_blocks(const uint8_t *block1, const uint8_t *block2,
                                   uint8_t *digest)
{
    uint32_t state[8];
    for (int i = 0; i < 8; i++)
        state[i] = SHA256_INIT[i];
    sha256_compress(block1, state);
    sha256_compress(block2, state);
    for (int i = 0; i < 8; i++)
        put_be32(digest + i * 4, state[i]);
}


__device__ __constant__ uint32_t RIPEMD160_K_LEFT[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};
__device__ __constant__ uint32_t RIPEMD160_K_RIGHT[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

__device__ __constant__ uint8_t RIPEMD160_R_LEFT[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
};
__device__ __constant__ uint8_t RIPEMD160_R_RIGHT[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
};
__device__ __constant__ uint8_t RIPEMD160_S_LEFT[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
};
__device__ __constant__ uint8_t RIPEMD160_S_RIGHT[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
};

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

__device__ __forceinline__ uint32_t rmd_f(int j, uint32_t x, uint32_t y, uint32_t z)
{
    if (j < 16)
        return x ^ y ^ z;
    if (j < 32)
        return (x & y) | (~x & z);
    if (j < 48)
        return (x | ~y) ^ z;
    if (j < 64)
        return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

/*
 * RIPEMD160 computation (input is 32-byte SHA256 digest)
 * input  : 32-byte input
 * digest : 20-byte output
 */
__device__ void ripemd160(const uint8_t *input, uint8_t *digest)
{
    /* Construct padded message (32-byte input -> 64-byte padded block) */
    uint32_t m[16];
    /* Little-endian read */
    for (int i = 0; i < 8; i++) {
        m[i] = ((uint32_t)input[i*4])       |
               ((uint32_t)input[i*4+1] << 8) |
               ((uint32_t)input[i*4+2] << 16)|
               ((uint32_t)input[i*4+3] << 24);
    }
    /* padding: 0x80 + zero fill + 64-bit bit length (little-endian) */
    m[8]  = 0x00000080;
    m[9]  = 0; m[10] = 0; m[11] = 0; m[12] = 0; m[13] = 0;
    m[14] = 256;  /* bit length = 32 * 8 = 256, low 32 bits */
    m[15] = 0;    /* high 32 bits */

    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476, h4 = 0xC3D2E1F0;

    uint32_t al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    uint32_t ar = h0, br = h1, cr = h2, dr = h3, er = h4;

    for (int j = 0; j < 80; j++) {
        int round = j / 16;
        uint32_t tl = ROTL32(al + rmd_f(j, bl, cl, dl) +
                             m[RIPEMD160_R_LEFT[j]] + RIPEMD160_K_LEFT[round],
                             RIPEMD160_S_LEFT[j]) + el;
        al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = tl;

        uint32_t tr = ROTL32(ar + rmd_f(79-j, br, cr, dr) +
                             m[RIPEMD160_R_RIGHT[j]] + RIPEMD160_K_RIGHT[round],
                             RIPEMD160_S_RIGHT[j]) + er;
        ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = tr;
    }

    uint32_t t = h1 + cl + dr;
    h1 = h2 + dl + er; h2 = h3 + el + ar;
    h3 = h4 + al + br; h4 = h0 + bl + cr; h0 = t;

    /* Little-endian output */
    put_le32(digest, h0);
    put_le32(digest + 4, h1);
    put_le32(digest + 8, h2);
    put_le32(digest + 12, h3);
    put_le32(digest + 16, h4);
}

/* GPU Kernel */

/*
 * Kernel: batch Hash160 computation
 * Optimization: each thread handles one (chain, step) pair to fully exploit GPU parallelism.
 * Total threads = num_chains * steps; each thread computes Hash160 for one public key.
 *
 * Input:
 *   aff_x, aff_y : affine coordinates (from gpu_secp256k1.cu)
 *   valid        : chain valid step counts
 * Output:
 *   hash160_comp   : [num_chains * steps * 20] bytes, compressed pubkey Hash160
 *   hash160_uncomp : [num_chains * steps * 20] bytes, uncompressed pubkey Hash160
 */
__global__ void kernel_hash160(
    const uint8_t * __restrict__ aff_x,
    const uint8_t * __restrict__ aff_y,
    const int     * __restrict__ valid,
    uint8_t       * __restrict__ hash160_comp,
    uint8_t       * __restrict__ hash160_uncomp,
    int num_chains,
    int steps)
{
    /* Each thread handles one (chain, step) pair */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chains * steps;
    if (tid >= total)
        return;

    int chain = tid / steps;
    int step  = tid % steps;

    /* Skip invalid steps */
    if (step >= valid[chain])
        return;

    int idx = chain * steps + step;
    const uint8_t *x = aff_x + idx * 32;
    const uint8_t *y = aff_y + idx * 32;

    /* ---- Construct compressed public key (33 bytes) and compute Hash160 ---- */
    uint8_t comp_block[64];
    /* Prefix: 0x02 (y even) or 0x03 (y odd) */
    comp_block[0] = (y[31] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++)
        comp_block[1 + i] = x[i];
    /* SHA256 padding for 33 bytes */
    comp_block[33] = 0x80;
    for (int i = 34; i < 56; i++)
        comp_block[i] = 0;
    /* bit length = 33 * 8 = 264 = 0x108, big-endian */
    comp_block[56] = 0;
    comp_block[57] = 0;
    comp_block[58] = 0;
    comp_block[59] = 0;
    comp_block[60] = 0;
    comp_block[61] = 0;
    comp_block[62] = 0x01;
    comp_block[63] = 0x08;

    uint8_t sha256_out[32];
    sha256_single_block(comp_block, sha256_out);
    ripemd160(sha256_out, hash160_comp + idx * 20);

    /* ---- Construct uncompressed public key (65 bytes) and compute Hash160 ---- */
    /* block1: 0x04 + x (32 bytes) + first 31 bytes of y = 64 bytes */
    uint8_t uncomp_block1[64];
    uint8_t uncomp_block2[64];
    uncomp_block1[0] = 0x04;
    for (int i = 0; i < 32; i++)
        uncomp_block1[1 + i] = x[i];
    for (int i = 0; i < 31; i++)
        uncomp_block1[33 + i] = y[i];

    /* block2: last 1 byte of y + padding + bit length */
    uncomp_block2[0] = y[31];
    uncomp_block2[1] = 0x80;
    for (int i = 2; i < 56; i++)
        uncomp_block2[i] = 0;
    /* bit length = 65 * 8 = 520 = 0x208, big-endian */
    uncomp_block2[56] = 0;
    uncomp_block2[57] = 0;
    uncomp_block2[58] = 0;
    uncomp_block2[59] = 0;
    uncomp_block2[60] = 0;
    uncomp_block2[61] = 0;
    uncomp_block2[62] = 0x02;
    uncomp_block2[63] = 0x08;

    sha256_two_blocks(uncomp_block1, uncomp_block2, sha256_out);
    ripemd160(sha256_out, hash160_uncomp + idx * 20);
}

/* Device Buffer Management */

static uint8_t *d_hash160_comp   = NULL;
static uint8_t *d_hash160_uncomp = NULL;
static int      g_h160_num_chains = 0;
static int      g_h160_steps     = 0;

int gpu_hash160_alloc(int num_chains, int steps)
{
    g_h160_num_chains = num_chains;
    g_h160_steps = steps;
    size_t sz = (size_t)num_chains * steps * 20;

    if (cudaMalloc(&d_hash160_comp, sz) != cudaSuccess)
        goto fail;
    if (cudaMalloc(&d_hash160_uncomp, sz) != cudaSuccess)
        goto fail;

    keylog_info("[GPU] Hash160 buffers allocated: %d chains x %d steps x 20 bytes x 2",
                num_chains, steps);
    return 0;
fail:
    keylog_error("[GPU] Hash160 buffer allocation failed");
    return -1;
}

void gpu_hash160_free(void)
{
    if (d_hash160_comp) {
        cudaFree(d_hash160_comp);
        d_hash160_comp = NULL;
    }
    if (d_hash160_uncomp) {
        cudaFree(d_hash160_uncomp);
        d_hash160_uncomp = NULL;
    }
}

/*
 * Execute Hash160 kernel (called after gpu_secp256k1_run)
 * aff_x, aff_y, valid : device pointers from gpu_secp256k1.cu
 * stream              : CUDA stream for async execution
 */
int gpu_hash160_run(const uint8_t *d_aff_x, const uint8_t *d_aff_y,
                    const int *d_valid, cudaStream_t stream)
{
    /* Each thread handles one (chain, step) pair; total threads = num_chains * steps */
    int block_size = 256;
    int total_threads = g_h160_num_chains * g_h160_steps;
    int grid_size = (total_threads + block_size - 1) / block_size;

    kernel_hash160<<<grid_size, block_size, 0, stream>>>(
        d_aff_x, d_aff_y, d_valid,
        d_hash160_comp, d_hash160_uncomp,
        g_h160_num_chains, g_h160_steps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        keylog_error("[GPU] kernel_hash160 execution failed: %s", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/* Return device-side Hash160 buffer pointers (for use by gpu_hashtable.cu) */
const uint8_t *gpu_hash160_get_comp(void)
{
    return d_hash160_comp;
}

const uint8_t *gpu_hash160_get_uncomp(void)
{
    return d_hash160_uncomp;
}

int gpu_hash160_get_num_chains(void)
{
    return g_h160_num_chains;
}

int gpu_hash160_get_steps(void)
{
    return g_h160_steps;
}
