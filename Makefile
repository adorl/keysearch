CC = gcc
CFLAGS = -O3 -g -ggdb
LDFLAGS = -lpthread

SECP256K1_SRC ?= ./secp256k1

# 检测内部头文件是否存在
SECP256K1_INTERNAL_HEADER = $(SECP256K1_SRC)/src/group.h
ifeq ($(wildcard $(SECP256K1_INTERNAL_HEADER)),)
    $(warning 警告：未找到$(SECP256K1_INTERNAL_HEADER)，回退到公开API模式
    CFLAGS += -DUSE_PUBKEY_API_ONLY
    LDFLAGS += -lsecp256k1
    SECP256K1_OBJ =
else
    CFLAGS += -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC)
    # 将secp256k1.c及预计算表编译为本地目标文件，不依赖系统库
    SECP256K1_OBJ = secp256k1_lib.o precomputed_ecmult.o precomputed_ecmult_gen.o
endif

# 源文件列表
SRCS_MAIN = keysearch.c hash_utils.c sha256.c ripemd160.c rand_key.c secp256k1_keygen.c
SRCS_TEST = test_case.c hash_utils.c sha256.c ripemd160.c rand_key.c secp256k1_keygen.c

OBJS_MAIN = $(SRCS_MAIN:.c=.o) $(SECP256K1_OBJ)
OBJS_TEST = $(SRCS_TEST:.c=.o) $(SECP256K1_OBJ)

.PHONY: all clean test

all: keysearch test

keysearch: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_case: $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# secp256k1 本地编译单元：secp256k1.c自己定义SECP256K1_BUILD
secp256k1_lib.o: $(SECP256K1_SRC)/src/secp256k1.c
	$(CC) $(CFLAGS) -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC) \
	    -c -o $@ $<

precomputed_ecmult.o: $(SECP256K1_SRC)/src/precomputed_ecmult.c
	$(CC) $(CFLAGS) -DSECP256K1_BUILD -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC) \
	    -c -o $@ $<

precomputed_ecmult_gen.o: $(SECP256K1_SRC)/src/precomputed_ecmult_gen.c
	$(CC) $(CFLAGS) -DSECP256K1_BUILD -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC) \
	    -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: test_case
	./test_case

clean:
	rm -f *.o keysearch test_case

