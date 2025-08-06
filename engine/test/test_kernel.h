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
#include <string>
#include <cuda_runtime.h>
#include "src/pack/bit_packing.h"
#include "common/timer.h"
#include "common/base.h"

#define TEST(X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE)      \
    {                                                                                          \
        std::cout << GPU_ARCH << " " << config_str << " ";                                     \
        printf("%d %d %d %d %d %d %d %d %d %d %d ", BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K,  \
               NSTAGE, CTA_TILE_STRIDE);                                                                        \
        int ret = benchmark<FQ_INIT_FUN(FQBMMA), FQ_EXEC_FUN(FQBMMA), FQ_OP_STATE(FQBMMA)>( \
            FQ_NAME_FUN(FQBMMA, Init, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
                        MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE),                                                 \
            FQ_NAME_FUN(FQBMMA, Exec, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
                        MMA_N, MMA_K, NSTAGE, CTA_TILE_STRIDE),                                                 \
            x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, d_x_scale, d_w_scale, m, n, k, d_out, h_out,      \
            h_ref_out, false, SIGNED, group_size, exec_dur, pack_dur, stream, warmup, repeat);             \
        if (ret == 0 && gflop_count / exec_dur > max_gflop) {                                  \
            max_gflop = gflop_count / exec_dur;                                                \
            best_config.str("");                                                               \
            best_config << BM << ", " << BN << ", " << BK << ", " << WM << ", " << WN << ", "  \
                        << WK << ", " << MMA_M << ", " << MMA_N << ", " << MMA_K << ", "       \
                        << NSTAGE << ", " << CTA_TILE_STRIDE;                                                             \
        }                                                                                      \
        printf("packing %f (us) exec %f (us) %f TOPS | %f B-TOPS | %s\n", pack_dur * 1e3,      \
               exec_dur * 1e3, gflop_count / exec_dur, true_gflop_count / exec_dur,            \
               ret == 0  ? "PASSED" :                                                          \
               ret == -1 ? "ERROR" :                                                           \
                           "FAILED");                                                          \
    }


inline bool check(const int *ref_out, const int *out, int m, int n)
{
    for (int i = 0; i < m * n; ++i) {
        if (ref_out[i] != out[i]) {
            return false;
        }
    }
    return true;
}

inline bool check(const half *ref_out, const half *out, int m, int n)
{
    for (int i = 0; i < m * n; ++i) {
        float dev = fabs(__half2float(ref_out[i]) - __half2float(out[i]));
        // Error tolerance within 0.01% of half precision range
        if (dev > 0.0001 * HALF_MAX_RANGE) {
            return false;
        }
    }
    return true;
}

/// benchmark func for bmma
template <typename InitFuncType, typename ExecFuncType, typename OpStateType>
inline int benchmark(InitFuncType init_fn, ExecFuncType exec_fn, int X_BITS, int W_BITS, int *X,
                     int *W, int *X_PACKED, int *W_PACKED, half* X_SCALE, half* W_SCALE, int M, int N, int K, half *D,
                     half *H_OUT, const half *H_REF_OUT, bool bias, bool SIGNED, int group_size, float &exec_dur,
                     float &pack_dur, cudaStream_t stream = NULL, int warmup = 10, int repeat = 100)
{
    auto w_pack_func = [&]() {
        if (W_BITS <= 32) {
            cudaError_t err = flexq_bit_packing(W, W_PACKED, N, K, W_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'weight launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport W_BITS %d: for launch_pack func \n", W_BITS);
            return -1;
        }
        return 0;
    };

    auto x_pack_func = [&]() {
        if (X_BITS <= 32) {
            cudaError_t err = flexq_bit_packing(X, X_PACKED, M, K, X_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'activation launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport X_BITS %d: for launch_pack func \n", X_BITS);
            return -1;
        }
        return 0;
    };

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }
    w_pack_func();
    x_pack_func();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }
    OpStateType state = (*init_fn)(X_PACKED, W_PACKED, X_SCALE, W_SCALE, M, N, K, D, group_size, false);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return -1;
    }
    (*exec_fn)(state, stream);
    cudaDeviceSynchronize();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }

    // profling exec func
    CudaTimer exec_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            exec_timer.start();
        (*exec_fn)(state, stream);
    }
    exec_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "exec kernel failed." << std::endl;
        return -1;
    }
    exec_dur = exec_timer.elapsed_msecs() / repeat;

    // profling packing func
    CudaTimer packing_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            packing_timer.start();
        x_pack_func();
    }
    packing_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "packing kernel failed." << std::endl;
        return -1;
    }
    pack_dur = packing_timer.elapsed_msecs() / repeat;

    // accuracy comparison
    cudaMemcpy(H_OUT, D, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    if (!check(H_REF_OUT, H_OUT, M, N)) {
        return -2;
    }
    return 0;
}


void test_w6a6_kernel(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, half* d_x_scale, half* d_w_scale, int m,
                    int n, int k, half *d_out, half *h_out, half *h_ref_out, int warmup, int repeat,
                    bool quant_sign, int group_size, cudaStream_t stream);

void test_w6a8_kernel(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, half* d_x_scale, half* d_w_scale, int m,
                    int n, int k, half *d_out, half *h_out, half *h_ref_out, int warmup, int repeat,
                    bool quant_sign, int group_size, cudaStream_t stream);