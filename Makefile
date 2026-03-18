CC = gcc
CFLAGS = -O3 -g -ggdb -march=native
LDFLAGS = -lpthread

ifdef GPU
    # Check if nvcc exists
    NVCC := $(shell which nvcc 2>/dev/null)
    ifeq ($(NVCC),)
        $(error Error: nvcc not found. Please install the CUDA toolkit, or build without GPU=1 for CPU-only mode)
    endif
    CFLAGS  += -DUSE_GPU
    NVCCFLAGS = -O3 -arch=sm_60 --compiler-options "-O3 -march=native"
    LDFLAGS += -lcuda -lcudart
    GPU_SRCS = gpu/gpu_detect.cu gpu/gpu_secp256k1.cu \
               gpu/gpu_hashtable.cu gpu/gpu_search.cu
    GPU_OBJS = $(GPU_SRCS:.cu=.o)
else
    GPU_OBJS =
endif

SECP256K1_SRC ?= ./secp256k1

# Check if internal header exists
SECP256K1_INTERNAL_HEADER = $(SECP256K1_SRC)/src/group.h
ifeq ($(wildcard $(SECP256K1_INTERNAL_HEADER)),)
    $(warning Warning: $(SECP256K1_INTERNAL_HEADER) not found, falling back to public API mode)
    CFLAGS += -DUSE_PUBKEY_API_ONLY
    LDFLAGS += -lsecp256k1
    SECP256K1_OBJ =
else
    CFLAGS += -I$(SECP256K1_SRC)/src -I$(SECP256K1_SRC)
    # Compile secp256k1.c and precomputed tables as local object files, no dependency on system library
    SECP256K1_OBJ = secp256k1_lib.o precomputed_ecmult.o precomputed_ecmult_gen.o
endif

# Detect CPU architecture: only compile AVX2 files on x86-64
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
    SIMD_SRCS = sha256_avx2.c ripemd160_avx2.c sha256_avx512.c ripemd160_avx512.c hash_utils_avx512.c secp256k1_keygen_avx512.c
else
    SIMD_SRCS =
endif

# Source file lists
SRCS_MAIN = keysearch.c keylog.c hash_utils.c sha256.c ripemd160.c $(SIMD_SRCS) rand_key.c secp256k1_keygen.c
SRCS_TEST = test_case.c test_gpu.c keylog.c hash_utils.c sha256.c ripemd160.c $(SIMD_SRCS) rand_key.c secp256k1_keygen.c

OBJS_MAIN = $(SRCS_MAIN:.c=.o) $(SECP256K1_OBJ) $(GPU_OBJS)
OBJS_TEST = $(SRCS_TEST:.c=.o) $(SECP256K1_OBJ)

.PHONY: all clean test

all: keysearch test

keysearch: $(OBJS_MAIN)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compilation rule for GPU .cu files (only active when GPU=1)
gpu/%.o: gpu/%.cu
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ $<

test_case: $(OBJS_TEST)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# secp256k1 local compilation unit: secp256k1.c defines SECP256K1_BUILD itself
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

# AVX2-specific files: compiled with -mavx2 (x86-64 only)
ifeq ($(ARCH),x86_64)
sha256_avx2.o: sha256_avx2.c
	$(CC) $(CFLAGS) -mavx2 -c -o $@ $<

ripemd160_avx2.o: ripemd160_avx2.c
	$(CC) $(CFLAGS) -mavx2 -c -o $@ $<

sha256_avx512.o: sha256_avx512.c
	$(CC) $(CFLAGS) -mavx512f -c -o $@ $<

ripemd160_avx512.o: ripemd160_avx512.c
	$(CC) $(CFLAGS) -mavx512f -c -o $@ $<

hash_utils.o: hash_utils.c
	$(CC) $(CFLAGS) -mavx2 -c -o $@ $<

hash_utils_avx512.o: hash_utils_avx512.c
	$(CC) $(CFLAGS) -mavx512f -mavx512ifma -mavx512bw -c -o $@ $<

secp256k1_keygen_avx512.o: secp256k1_keygen_avx512.c
	$(CC) $(CFLAGS) -mavx512f -mavx512ifma -c -o $@ $<

keysearch.o: keysearch.c
	$(CC) $(CFLAGS) -mavx512f -mavx512ifma -c -o $@ $<

test_case.o: test_case.c
	$(CC) $(CFLAGS) -mavx2 -mavx512f -mavx512ifma -c -o $@ $<
endif

test: test_case
	./test_case

clean:
	rm -f *.o gpu/*.o keysearch test_case
