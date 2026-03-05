#ifndef SECP256K1_KEYGEN_H
#define SECP256K1_KEYGEN_H

#include <stdint.h>
#include <stddef.h>

/*
 * secp256k1_keygen.h
 *
 * 提供：
 *   1. 直接点加法
 *   2. 批量仿射坐标归一化（Batch Normalization）
 *   3. 直接从仿射坐标构造公钥字节（跳过serialize）
 *
 * 编译模式：
 *   - 默认：引入libsecp256k1内部头文件
 *   - USE_PUBKEY_API_ONLY：回退到公开API
 */

#ifndef USE_PUBKEY_API_ONLY

/*
 * 内部接口模式：直接使用libsecp256k1内部类型和实现。
 * 编译时需要 -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC)
 * 链接时需要本地编译的secp256k1_lib.o
 * 以及precomputed_ecmult.o和precomputed_ecmult_gen.o
 */

/* 必须在包含任何secp256k1头文件之前定义SECP256K1_BUILD */
#ifndef SECP256K1_BUILD
#  define SECP256K1_BUILD
#endif

#include "../include/secp256k1.h"

/* 引入内部类型声明和static函数实现 */
#include "assumptions.h"
#include "util.h"
#include "field.h"
#include "field_impl.h"
#include "scalar.h"
#include "scalar_impl.h"
#include "group.h"
#include "group_impl.h"
#include "ecmult_gen.h"
#include "ecmult_gen_impl.h"
#include "int128_impl.h"
/*
 * 初始化全局生成元G的仿射坐标
 * 必须在所有线程启动前调用一次
 * 返回值：0 成功，-1 失败（infinity标志异常）
 */
int keygen_init_generator(const secp256k1_context *ctx,
                          secp256k1_ge *G_out);

/*
 * 从32字节私钥生成Jacobian坐标公钥（secp256k1_gej）
 * 内部调用secp256k1_ecmult_gen，仅在每批开始时调用一次
 * 返回值：0 成功，-1 失败
 */
int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_gej *gej_out);

/*
 * 批量将Jacobian坐标数组归一化为仿射坐标数组
 * 使用Montgomery trick：仅需1次模逆+3*(n-1)次乘法。
 * infinity点将被跳过（对应ge_out[i].infinity == 1）
 * 参数：
 *   gej_in  : 输入Jacobian坐标数组
 *   ge_out  : 输出仿射坐标数组（调用方分配，大小>=n）
 *   n       : 数组元素个数
 */
void keygen_batch_normalize(const secp256k1_gej *gej_in,
                            secp256k1_ge *ge_out,
                            size_t n);

/*
 * 利用rzr增量因子加速的批量归一化（省去前向累积对gej.z的内存读取）
 * rzr[i] 满足：Z[i+1] = Z[i] * rzr[i]（由secp256k1_gej_add_ge_var的rzr参数提供）
 * 要求：所有点均非infinity（内层循环保证），rzr数组大小为n-1
 * 参数：
 *   gej_in  : 输入Jacobian坐标数组（大小n）
 *   ge_out  : 输出仿射坐标数组（调用方分配，大小>=n）
 *   rzr     : Z坐标增量因子数组（大小n-1，rzr[i]对应gej_in[i]→gej_in[i+1]的Z增量）
 *   n       : 数组元素个数
 */
void keygen_batch_normalize_rzr(const secp256k1_gej *gej_in,
                                secp256k1_ge *ge_out,
                                const secp256k1_fe *rzr,
                                size_t n);

/*
 * 从仿射坐标直接构造压缩/非压缩公钥字节，跳过serialize调用
 * 调用前必须确保ge->infinity == 0
 * 参数：
 *   ge              : 已归一化的仿射坐标点
 *   compressed_out  : 压缩公钥输出（33字节），传NULL则跳过
 *   uncompressed_out: 非压缩公钥输出（65字节），传NULL则跳过
 */
void keygen_ge_to_pubkey_bytes(const secp256k1_ge *ge,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out);

#else

/* 回退模式下使用系统安装的secp256k1.h提供公开类型 */
#include <secp256k1.h>

/*
 * 回退模式：初始化（无操作，仅做兼容）
 * 返回值：0 成功
 */
int keygen_init_generator(const secp256k1_context *ctx,
                          void *G_out);

/*
 * 回退模式：从私钥创建公钥（使用secp256k1_ec_pubkey_create）
 * gej_out实际为secp256k1_pubkey*
 * 返回值：0 成功，-1 失败
 */
int keygen_privkey_to_gej(const secp256k1_context *ctx,
                          const uint8_t privkey[32],
                          secp256k1_pubkey *pubkey_out);

/*
 * 回退模式：单个公钥序列化为压缩/非压缩字节
 * 参数：
 *   pubkey          : secp256k1_pubkey指针
 *   compressed_out  : 压缩公钥输出（33字节），传NULL则跳过
 *   uncompressed_out: 非压缩公钥输出（65字节），传NULL则跳过
 */
void keygen_ge_to_pubkey_bytes(const secp256k1_context *ctx,
                               const secp256k1_pubkey *pubkey,
                               uint8_t *compressed_out,
                               uint8_t *uncompressed_out);

#endif /* USE_PUBKEY_API_ONLY */

#endif /* SECP256K1_KEYGEN_H */

