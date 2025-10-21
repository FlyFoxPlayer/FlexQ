/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <assert.h>
#define BITS_INT 32
#define BITS_INT4 128
#define INT4_NUIT 4

#define WARP_SIZE 32
#define WARP_M 1
#define WARP_K 32
#define MMA_M 8
// only support group size = 128
#define GROUP_SIZE 128
#define THREADS_NUM 512
#define WARP_PER_BLOCK (THREADS_NUM / WARP_SIZE) 
#define M_WARP_NUM 1
#define K_WARP_NUM 16
#define BLOCK_M (WARP_M * M_WARP_NUM)
#define BLOCK_K (WARP_K * K_WARP_NUM)

#define SCALE_PACKING_A(x) ((x) * 2)
#define SCALE_PACKING_B(x) ((x) / 2)

#define SCALE_SIZE_X(x) ((x + 3) / 4 * 4)  // Align to 16 bytes 
#define SCALE_SIZE_W(x) (x / 2)  // Align to 16 bytes (x is usually BLOCK_N, ensuring 8-byte alignment)

//how many bits steps to go
#define STEP4(X) (((X)+3)>>2) 
#define STEP8(X) (((X)+7)>>3) 
#define STEP16(X) (((X)+15)>>4) 
#define STEP32(X) (((X)+31)>>5) 
#define STEP64(X) (((X)+63)>>6) 
#define STEP128(X) (((X)+127)>>7) 
#define STEP_Y(X, Y) (((X)+(Y-1))>>(31 - __builtin_clz(Y)))
//total bits covers after padding
#define PAD4(X) (STEP4(X)<<2)
#define PAD8(X) (STEP8(X)<<3)
#define PAD16(X) (STEP16(X)<<4)
#define PAD32(X) (STEP32(X)<<5)
#define PAD64(X) (STEP64(X)<<6)
#define PAD128(X) (STEP128(X)<<7)
#define PAD_Y(X, Y) (STEP_Y(X, Y)<<(31 - __builtin_clz(Y)))

namespace fastertransformer {

// clang-format off
template<typename T> struct GeluActivation;
template<typename T> struct ReluActivation;
template<typename T> struct SiluActivation;
template<typename T> struct IdentityActivation;
// clang-format on

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(int*         packed_out,
                             T*           out,
                             const BT*    bias,
                             const T*     gated_weights,
                             const BT*    gated_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             half*        output_scale,
                             int          quant_bits,
                             const int*   padding_offset,
                             const int    seq_len,
                             cudaStream_t stream);

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericActivation(T*           out,
                             const BT*    bias,
                             const T*     gated_weights,
                             const BT*    gated_bias,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int    m,
                             const int    n,
                             const int    int8_mode,
                             const float* activation_in,
                             const float* activation_out,
                             cudaStream_t stream)
{
    invokeGenericActivation<Activation, T, BT>(nullptr,
                                               out,
                                               bias,
                                               gated_weights,
                                               gated_bias,
                                               ia3_tasks,
                                               ia3_weights,
                                               m,
                                               n,
                                               int8_mode,
                                               activation_in,
                                               activation_out,
                                               (half*)nullptr,
                                               0,
                                               (const int*)nullptr,
                                               0,
                                               stream);
}

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericMaskActivation(T*           out0,
                                 T*           out1,
                                 int*           mask,
                                 const BT*    bias0,
                                 const BT*    bias1,
                                 const T*     gated_weights,
                                 const BT*    gated_bias,
                                 const int*   ia3_tasks,
                                 const T*     ia3_weights,
                                 const int    m,
                                 const int    n,
                                 const int    int8_mode,
                                 const float* activation_in,
                                 const float* activation_out,
                                 const int*   padding_offset,
                                 const int    seq_len,
                                 cudaStream_t stream);

template<template<typename T> class Activation, typename T, typename BT>
void invokeGenericMaskActivation(T*           out0,
                                 T*           out1,
                                 int*           mask,
                                 const BT*    bias0,
                                 const BT*    bias1,
                                 const T*     gated_weights,
                                 const BT*    gated_bias,
                                 const int*   ia3_tasks,
                                 const T*     ia3_weights,
                                 const int    m,
                                 const int    n,
                                 const int    int8_mode,
                                 const float* activation_in,
                                 const float* activation_out,
                                 cudaStream_t stream)
{
    invokeGenericMaskActivation<Activation, T, BT>(out0,
                                               out1,
                                               mask,
                                               bias0,
                                               bias1,
                                               gated_weights,
                                               gated_bias,
                                               ia3_tasks,
                                               ia3_weights,
                                               m,
                                               n,
                                               int8_mode,
                                               activation_in,
                                               activation_out,
                                               (const int*)nullptr,
                                               0,
                                               stream);
}

template<typename T>
void invokeAddBiasGeluV2(T*           out,
                         const T*     bias,
                         const int*   ia3_tasks,
                         const T*     ia3_weights,
                         const int*   padding_offset,
                         const int    seq_len,
                         const int    m,
                         const int    n,
                         cudaStream_t stream);

template<typename T>
void invokeLoraMaskAddBiasGeluV2(T*           out0,
                             T*           out1,
                             int*         mask,
                             const T*     bias0,
                             const T*     bias1,
                             const int*   ia3_tasks,
                             const T*     ia3_weights,
                             const int*   padding_offset,
                             const int    seq_len,
                             const int    m,
                             const int    n,
                             cudaStream_t stream);

template<typename T>
void invokeAddBias(T* out, T const* bias, const int m, const int n, cudaStream_t stream)
{
    invokeGenericActivation<IdentityActivation, T, T>(
        out, bias, nullptr, nullptr, nullptr, nullptr, m, n, 0, nullptr, nullptr, stream);
}

template<typename T>
void invokeAddBiasGeluV2(
    T* out, const T* bias, const int* ia3_tasks, const T* ia3_weights, const int m, const int n, cudaStream_t stream)
{
    invokeAddBiasGeluV2(out, bias, ia3_tasks, ia3_weights, nullptr, 0, m, n, stream);
}

template<typename T>
void invokeLoraMaskAddBiasGeluV2(
    T* out0, T* out1, int* mask, const T* bias0, const T* bias1, const int* ia3_tasks, const T* ia3_weights, const int m, const int n, cudaStream_t stream)
{
    invokeLoraMaskAddBiasGeluV2(out0, out1, mask, bias0, bias1, ia3_tasks, ia3_weights, nullptr, 0, m, n, stream);
}

template<typename T>
void invokeAddBiasTanh(T* out, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeSigmoid(T* data, const int size, const float scale, cudaStream_t stream);

// FlexQ adapter
template<typename Ti1, typename Ti2, typename To> __device__ inline To get_T_max(Ti1 val1, Ti2 val2) { return (val1 > val2) ? val1 : val2; }

__device__ inline half get_T_max(half val1, half2 val2) {
    val1 = (val1 > val2.x ? val1 : val2.x);
    return (val1 > val2.y ? val1 : val2.y);
}
__device__ inline half get_T_max(half val1, float val2) {
    return (val1 > __float2half(val2)) ? val1 : __float2half(val2); 
}
__device__ inline half get_T_max(half val1, __nv_bfloat162 val2) {
    val1 = (val1 > __float2half((float)val2.x) ? val1 : __float2half((float)val2.x));
    return (val1 > __float2half((float)val2.y) ? val1 : __float2half((float)val2.y));
}

template<typename Ti, typename To> __device__ inline To pack_input_frag(Ti val) { return val; }
__device__ inline float2 pack_input_frag(float val) {
    float2 result;
    result.x = val;
    result.y = val;
    return result;
}
__device__ inline float2 pack_input_frag(half2 val) { 
    float2 result;
    result.x = __half2float(val.x);
    result.y = __half2float(val.y);
    return result;
}
__device__ inline float2 pack_input_frag(__nv_bfloat162 val) { 
    float2 result;
    result.x = (float)val.x;
    result.y = (float)val.y;
    return result;
}

}  // namespace fastertransformer
