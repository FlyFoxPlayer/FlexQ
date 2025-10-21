#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

class FLEXQGEMMWrapper {
private:
    int           x_bits_;
    int           w_bits_;
    bool          signed_;

public:
    FLEXQGEMMWrapper(int X_BITS, int W_BITS, bool SIGNED);
    ~FLEXQGEMMWrapper();

    void pack(const half* in_data, int* packed_data, half* x_scale, int M, int K, int BIT, cudaStream_t stream);
    void gemm(const int    M,
              const int    N,
              const int    K,
              const int*   A,
              const int*   B,
              const half*  C,
              half*        D,
              float*       x_scale,
              const float* w_scale,
              const float* scale_inter,
              const float* scale_out,
              bool         bias,
              char*        flexq_gemm_workspace,
              size_t       flexq_gemm_ws_bytes,
              cudaStream_t stream);

    void gemm(const int    M,
              const int    N,
              const int    K,
              const half*  A,
              const int*   B,
              const half*  C,
              half*        D,
              float*       x_scale,
              const float* w_scale,
              const float* scale_inter,
              const float* scale_out,
              bool         bias,
              char*        flexq_gemm_workspace,
              size_t       flexq_gemm_ws_bytes,
              cudaStream_t stream);
};
