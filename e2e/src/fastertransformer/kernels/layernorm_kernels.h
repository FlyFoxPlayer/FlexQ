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
#include "src/fastertransformer/utils/cuda_utils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

// how many bits steps to go
#define STEP4(X) (((X)+3)>>2) 
#define STEP8(X) (((X)+7)>>3) 
#define STEP16(X) (((X)+15)>>4) 
#define STEP32(X) (((X)+31)>>5) 
#define STEP64(X) (((X)+63)>>6) 
#define STEP128(X) (((X)+127)>>7) 
#define STEP_Y(X, Y) (((X)+(Y-1))>>(31 - __builtin_clz(Y)))

// total bits covers after padding
#define PAD4(X) (STEP4(X)<<2)
#define PAD8(X) (STEP8(X)<<3)
#define PAD16(X) (STEP16(X)<<4)
#define PAD32(X) (STEP32(X)<<5)
#define PAD64(X) (STEP64(X)<<6)
#define PAD128(X) (STEP128(X)<<7)
#define PAD_Y(X, Y) (STEP_Y(X, Y)<<(31 - __builtin_clz(Y)))

#define SCALE_PACKING_A(x) ((x) * 2)
#define SCALE_PACKING_B(x) ((x) / 2)

#define SCALE_SIZE_X(x) ((x + 3) / 4 * 4)  // Align to 16 bytes 
#define SCALE_SIZE_W(x) (x / 2)  // Align to 16 bytes (x is usually BLOCK_N, ensuring 8-byte alignment)

namespace fastertransformer {

enum class LayerNormType {
    pre_layernorm,
    post_layernorm,
    InvalidType
};

inline LayerNormType getLayerNormType(std::string layernorm_type_str)
{
    if (layernorm_type_str == "pre_layernorm") {
        return LayerNormType::pre_layernorm;
    }
    else if (layernorm_type_str == "post_layernorm") {
        return LayerNormType::post_layernorm;
    }
    else {
        FT_CHECK_WITH_INFO(false, "Layernorm Type: " + layernorm_type_str + " not supported !");
    }
    return LayerNormType::InvalidType;
}

template<typename T>
struct LayerNormWeight {
    const T* gamma = nullptr;
    const T* beta  = nullptr;
};

template<typename T>
void invokeAddBiasResidualLayerNorm(T*           out,
                                    const T*     input,
                                    const T*     bias,
                                    const T*     gamma,
                                    const T*     beta,
                                    const float  layernorm_eps,
                                    const int    m,
                                    const int    n,
                                    cudaStream_t stream);

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(int*         normed_packed_out,
                                              T*           output,
                                              T*           norm_output,
                                              const T*     input,
                                              const T*     residual1,
                                              const T*     residual2,
                                              const T*     gamma,
                                              const T*     beta,
                                              const T*     bias,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* scale_inter,
                                              const float* scale_out,
                                              float*       scale,
                                              float*       dynamic_scale,
                                              half*        norm_output_scale,
                                              const int    int8_mode,
                                              cudaStream_t stream,
                                              int          opt_version = 2);

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,
                                              T*           norm_output,
                                              const T*     input,
                                              const T*     residual1,
                                              const T*     gamma,
                                              const T*     beta,
                                              const T*     bias,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* scale_inter,
                                              const float* scale_out,
                                              float*       scale,
                                              float*       dynamic_scale,
                                              const int    int8_mode,
                                              cudaStream_t stream,
                                              half*        norm_output_scale = nullptr,
                                              int          opt_version = 2)
{
    invokeGeneralAddBiasResidualPreLayerNorm(nullptr,
                                             output,
                                             norm_output,
                                             input,
                                             residual1,
                                             (const T*)nullptr,
                                             gamma,
                                             beta,
                                             bias,
                                             layernorm_eps,
                                             m,
                                             n,
                                             scale_inter,
                                             scale_out,
                                             scale,
                                             dynamic_scale,
                                             norm_output_scale,
                                             int8_mode,
                                             stream,
                                             opt_version);
}

template<typename T>
void invokeGeneralAddBiasResidualPreLayerNorm(T*           output,
                                              T*           norm_output,
                                              const T*     input,
                                              const T*     residual1,
                                              const T*     gamma,
                                              const T*     beta,
                                              const T*     bias,
                                              const float  layernorm_eps,
                                              int          m,
                                              int          n,
                                              const float* scale_inter,
                                              const float* scale_out,
                                              float*       scale,
                                              float*       dynamic_scale,
                                              const int    int8_mode,
                                              cudaStream_t stream,
                                              int          opt_version,
                                              half* norm_output_scale = nullptr)
{
    invokeGeneralAddBiasResidualPreLayerNorm(nullptr,
                                             output,
                                             norm_output,
                                             input,
                                             residual1,
                                             (const T*)nullptr,
                                             gamma,
                                             beta,
                                             bias,
                                             layernorm_eps,
                                             m,
                                             n,
                                             scale_inter,
                                             scale_out,
                                             scale,
                                             dynamic_scale,
                                             norm_output_scale,
                                             int8_mode,
                                             stream,
                                             opt_version);
}

template<typename T>
void invokeGeneralLayerNorm(int*         normed_packed_out,
                            T*           out,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            const int    m,
                            const int    n,
                            float*       scale,
                            float*       dynamic_scale,
                            half*        norm_output_scale,
                            const int    int8_mode,
                            cudaStream_t stream,
                            int          opt_version = 2);

template<typename T>
void invokeGeneralLayerNorm(T*           out,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            const int    m,
                            const int    n,
                            float*       scale,
                            const int    int8_mode,
                            cudaStream_t stream,
                            half*        norm_output_scale = nullptr,
                            int          opt_version = 2)
{
    invokeGeneralLayerNorm(
        nullptr, out, input, gamma, beta, layernorm_eps, m, n, scale, (float*)nullptr, norm_output_scale, int8_mode, stream, opt_version = 2);
}

template<typename T>
void invokeGeneralLayerNorm(int*         normed_packed_out,
                            T*           out,
                            const T*     input,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            const int    m,
                            const int    n,
                            float*       scale,
                            const int    int8_mode,
                            cudaStream_t stream,
                            half*        norm_output_scale = nullptr,
                            int          opt_version = 2)
{
    invokeGeneralLayerNorm(
        normed_packed_out, out, input, gamma, beta, layernorm_eps, m, n, scale, (float*)nullptr, norm_output_scale, int8_mode, stream, opt_version = 2);
}

template<typename T>
void invokeGeneralT5LayerNorm(int*         normed_packed_out,
                              T*           out,
                              const T*     input,
                              const T*     gamma,
                              const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n,
                              half*        norm_output_scale,
                              cudaStream_t stream);

template<typename T>
void invokeGeneralT5LayerNorm(T*           out,
                              const T*     input,
                              const T*     gamma,
                              const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n,
                              cudaStream_t stream,
                              half*        norm_output_scale = nullptr)
{
    invokeGeneralT5LayerNorm(nullptr, out, input, gamma, beta, layernorm_eps, m, n, norm_output_scale, stream);
}

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(T*           output,
                                            T*           norm_output,
                                            int*         normed_packed_out,
                                            const T*     input,
                                            const T*     gamma,
                                            const float  layernorm_eps,
                                            int          m,
                                            int          n,
                                            float*       scale,
                                            float*       dynamic_scale,
                                            half*        norm_output_scale,
                                            const int    int8_mode,
                                            cudaStream_t stream);

template<typename T>
void invokeGeneralAddResidualT5PreLayerNorm(T*           output,
                                            T*           norm_output,
                                            const T*     input,
                                            const T*     gamma,
                                            const float  layernorm_eps,
                                            int          m,
                                            int          n,
                                            float*       scale,
                                            const int    int8_mode,
                                            cudaStream_t stream,
                                            half*        norm_output_scale = nullptr)
{
    invokeGeneralAddResidualT5PreLayerNorm(output, norm_output, nullptr, input, gamma, layernorm_eps, m, n, scale, (float*)nullptr, norm_output_scale, int8_mode, stream);
}

template<typename T>
void invokeGeneralAddBiasResidualT5PreLayerNorm(T*           output,
                                                T*           norm_output,
                                                const T*     input,
                                                const T*     gamma,
                                                const T*     beta,
                                                const T*     bias,
                                                const float  layernorm_eps,
                                                int          m,
                                                int          n,
                                                cudaStream_t stream);

template<typename T>
void invokeLayernormShiftPartition(T*           out,
                                   const T*     input,
                                   const T*     gamma,
                                   const T*     beta,
                                   const float  layernorm_eps,
                                   int          batch,
                                   int          H,
                                   int          W,
                                   int          n,
                                   int          shift_size,
                                   int          window_size,
                                   cudaStream_t stream);

template<typename T>
void invokeAddBiasLayernorm(T*           out,
                            const T*     bias,
                            const T*     gamma,
                            const T*     beta,
                            const float  layernorm_eps,
                            int          m,
                            int          n,
                            cudaStream_t stream,
                            int          opt_version = 2);

template<typename T>
void invokeMergeLayernorm(T*           output,
                          const T*     input,
                          const T*     gamma,
                          const T*     beta,
                          const float  layernorm_eps,
                          int          batch,
                          int          H,
                          int          W,
                          int          n,
                          cudaStream_t stream);

template<typename T>
void invokeAddBiasLayernormAddRes(T*           out,
                                  const T*     input,
                                  const T*     bias,
                                  const T*     gamma,
                                  const T*     beta,
                                  const float  layernorm_eps,
                                  int          m,
                                  int          n,
                                  cudaStream_t stream);
}  // namespace fastertransformer
