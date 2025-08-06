// Copyright (C) 2024 ByteDance and/or its affiliates
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//          http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cassert>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Define row/column major flag
#define ROW_FIRST 0
#define COL_FIRST 1

#define BITS_INT 32     // Number of bits in an int 
#define BITS_INT4 128   // Number of bits in an int4
#define INT4_NUIT 4     // Number of int4 elements 

#define WARP_SIZE 32

#define CEIL(x, y) (((x) + (y) - 1) / (y))

#define SCALE_PACKING_A(x) ((x) * 2)
#define SCALE_PACKING_B(x) ((x) / 2)

#define SCALE_SIZE_X(x) ((x + 3) / 4 * 4)  // Align to 16 bytes 
#define SCALE_SIZE_W(x) (x / 2)  // Align to 16 bytes (x is usually BLOCK_N, ensuring 8-byte alignment)

#define HALF_MAX_RANGE 65504

// How many bits steps to go
#define STEP4(X) (((X)+3)>>2) 
#define STEP8(X) (((X)+7)>>3) 
#define STEP16(X) (((X)+15)>>4) 
#define STEP32(X) (((X)+31)>>5) 
#define STEP64(X) (((X)+63)>>6) 
#define STEP128(X) (((X)+127)>>7) 
#define STEP_Y(X, Y) (((X)+(Y-1))>>(31 - __builtin_clz(Y)))
// Total bits covers after padding
#define PAD4(X) (STEP4(X)<<2)
#define PAD8(X) (STEP8(X)<<3)
#define PAD16(X) (STEP16(X)<<4)
#define PAD32(X) (STEP32(X)<<5)
#define PAD64(X) (STEP64(X)<<6)
#define PAD128(X) (STEP128(X)<<7)
#define PAD_Y(X, Y) (STEP_Y(X, Y)<<(31 - __builtin_clz(Y)))

// Macro to declare a device-side function
#define DEVICE_INLINE __device__ __forceinline__

#define ASSEMBLY asm volatile

#define CHECK(exp)                                                           \
    exp;                                                                     \
    const cudaError_t error = cudaGetLastError();                            \
    if (error != cudaSuccess) {                                              \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
        exit(1);                                                             \
    }  

inline bool isCudaSuccess(cudaError_t status)
{
    cudaError_t error = status;
    if (error != cudaSuccess) {
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// Compile-time log2 calculation (constexpr function)
constexpr int log2_constexpr(int k) {
    int log2k = 0;
    while (k >>= 1) ++log2k;
    return log2k;
}

constexpr int constexpr_min(int a, int b) {
    return a < b ? a : b;
}


template <int M_, int N_, int K_> struct ShapeBase {
    static constexpr int M = M_, N = N_, K = K_;
};

template <int X_BITS_, int W_BITS_, bool SIGNED_> struct QuantType {
    static constexpr int X_BITS = X_BITS_, W_BITS = W_BITS_;
    static constexpr bool SIGNED = SIGNED_;
};

template <uint32_t S_, uint32_t B_, uint32_t M_> struct SwizzleConfig {
    static constexpr uint32_t S = S_, B = B_, M = M_;
};


struct SwizzleIdentity {
    DEVICE_INLINE
    int operator()(int offset)
    {
        return offset;
    }
};

struct Swizzle8BWiseXor {
    DEVICE_INLINE
    int operator()(int offset)
    {
        return (offset ^ ((offset & (7 << 6)) >> 3));
    }
};

struct Swizzle6BWiseXor {
    DEVICE_INLINE
    int operator()(int offset)
    {
        if((offset / 3 / 32) & 1){
            return offset ^ (1 << 2);
        }
        return offset;
    }
};

// Bank conflict -- swizzle template  
/*  
    Address composition: [row index bits, column index bits, number of elements per swizzle unit]  
    Example: FP16 16x16 matrix (8 elements per unit): address bits → XXXX X XXX  

    S: SShift, bit distance from addr offset to BShift  
    B: BShift, number of columns = 2 ^ B  
    M: MBase, number of elements per swizzle unit (original elements read per thread)  

    Note: Ensure S > B (i.e., SShift > BShift)  
*/ 
template <typename SwizzleConfig>
struct Swizzle {
    DEVICE_INLINE
    uint32_t operator()(uint32_t addr){
        constexpr auto Bmask = ((1 << SwizzleConfig::B) - 1) << SwizzleConfig::M;
        return ((addr >> SwizzleConfig::S) & Bmask ) ^ addr;
    }
};




template <int NStage, bool UseMinSync> struct Pipeline;

template <int NStage> struct Pipeline<NStage, false> {
    DEVICE_INLINE
    void acquireWriter()
    {
    }
    DEVICE_INLINE
    void commitStage()
    {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    DEVICE_INLINE
    void acquireReader()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NStage - 1));
        __syncthreads();
    }
    DEVICE_INLINE
    void releaseReader()
    {
        __syncthreads();
    }
};

template <int NStage> struct Pipeline<NStage, true> {
    int ahead_stage = 0;
    DEVICE_INLINE
    void acquireWriter()
    {
        if (ahead_stage == NStage - 1) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NStage - 2));
            __syncthreads();
        }
    }
    DEVICE_INLINE
    void commitStage()
    {
        asm volatile("cp.async.commit_group;\n" ::);
        ahead_stage++;
    }
    DEVICE_INLINE
    void acquireReader()
    {
    }
    DEVICE_INLINE
    void releaseReader()
    {
        ahead_stage--;
    }
};

template <int N> struct IntVector;

template <> struct IntVector<4> {
    int x[4] = { 0, 0, 0, 0 };
    DEVICE_INLINE
    void ld(const int *src)
    {
        *(int4 *)x = *(int4 *)src;
    }
    DEVICE_INLINE
    void st(int *dst)
    {
        *(int4 *)dst = *(int4 *)x;
    }
    DEVICE_INLINE
    void reset()
    {
        x[0] = 0;
        x[1] = 0;
        x[2] = 0;
        x[3] = 0;
    }
};

template <int N> struct FloatVector;

template <> struct FloatVector<4> {
    float x[4] = { 0, 0, 0, 0 };
    DEVICE_INLINE
    void ld(const int *src)
    {
        *(float4 *)x = *(float4 *)src;
    }
    DEVICE_INLINE
    void st(int *dst)
    {
        *(float4 *)dst = *(float4 *)x;
    }
    DEVICE_INLINE
    void reset()
    {
        x[0] = 0.0f;
        x[1] = 0.0f;
        x[2] = 0.0f;
        x[3] = 0.0f;
    }
};

template <> struct FloatVector<1> {
    float x[1] = {0.0f};
    DEVICE_INLINE
    void ld(const int *src)
    {
        *(float *)x = *(float *)src;
    }
    DEVICE_INLINE
    void st(int *dst)
    {
        *(float *)dst = *(float *)x;
    }
    DEVICE_INLINE
    void reset()
    {
        x[0] = 0.0f;
    }
};

////////
#define FQ_INIT_FUN(type) type##InitFn_t

#define FQ_EXEC_FUN(type) type##ExecFn_t

#define FQ_OP_STATE(type) type##OpState

#define FQ_NAME_FUN(type, fn, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, \
                    WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE)                                         \
    type##_##X_BITS##x##W_BITS##x##SIGNED##_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_##WARP_M##x##WARP_N##x##WARP_K##_##MMA_M##x##MMA_N##x##MMA_K##_##NSTAGE##_##CTA_TILE_STRIDE##_##fn##Fn

#define FQ_INSTANTIATE_FUN(type, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M,      \
                           WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE)                          \
    type##InitFn_t FQ_NAME_FUN(type, Init, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K,    \
                               WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE) =            \
        type##InitFn<QuantType<X_BITS, W_BITS, SIGNED>, ShapeBase<BLOCK_M, BLOCK_N, BLOCK_K>,    \
                     ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE, CTA_TILE_STRIDE>; \
    type##ExecFn_t FQ_NAME_FUN(type, Exec, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K,    \
                               WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE) =            \
        type##ExecFn<QuantType<X_BITS, W_BITS, SIGNED>, ShapeBase<BLOCK_M, BLOCK_N, BLOCK_K>,    \
                     ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE, CTA_TILE_STRIDE>;

#define FQ_DECL_FUN(type, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, \
                    WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE)                                     \
    extern type##InitFn_t FQ_NAME_FUN(type, Init, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N,  \
                                      BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K,  \
                                      NSTAGE, CTA_TILE_STRIDE);                                               \
    extern type##ExecFn_t FQ_NAME_FUN(type, Exec, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N,  \
                                      BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K,  \
                                      NSTAGE, CTA_TILE_STRIDE);