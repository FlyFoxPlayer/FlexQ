
#include "src/fastertransformer/kernels/flexqgemm/flexq_gemm_wrapper.h"
#include "src/fastertransformer/kernels/flexqgemm/src/bgemm/flexq_bmma_library.h"
#include "src/fastertransformer/kernels/flexqgemm/src/pack/bit_packing.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


FLEXQGEMMWrapper::FLEXQGEMMWrapper(int X_BITS, int W_BITS, bool SIGNED): x_bits_(X_BITS), w_bits_(W_BITS), signed_(SIGNED)
{
}

FLEXQGEMMWrapper::~FLEXQGEMMWrapper() {}

void FLEXQGEMMWrapper::pack(const half* in_data, int* packed_data, half* x_scale, int M, int K, int BIT, cudaStream_t stream){
    flexq_bit_packing(in_data, packed_data, x_scale, M, K, BIT, stream);
}

void FLEXQGEMMWrapper::gemm(const int    M,
                          const int    N,
                          const int    K,
                          const int*   A,
                          const int*   B,
                          const half*  C,
                          half*        D,
                          float* x_scale,
                          const float* w_scale,
                          const float* scale_inter,
                          const float* scale_out,
                          bool         bias,
                          char*        flexq_gemm_workspace,
                          size_t       flexq_gemm_ws_bytes,
                          cudaStream_t stream = NULL)
{
    half *x_scale_half = reinterpret_cast<half*>(x_scale);
    const half *w_scale_half = reinterpret_cast<const half*>(w_scale);

    FQBMMAInitFn_t init_fn;
    FQBMMAExecFn_t exec_fn;
    FQBMMAOpState  state;
    if (K < 128 || K % 128 != 0) {
        printf("[FlexQ][Error] unsupport K = %d\n", K);
        return;
    }
    /*
        The optimal FlexQ kernel layout configuration for the LLaMA model on the A6000-48GB.
        If switching models, the layout configuration should be manually replaced for better performance (located at the end of this file).
        If changing hardware, select the optimal layout configuration based on kernel benchmark.
    */
    // w6a6
    if (w_bits_ == 6 && x_bits_ == 6) {
        if(M == 1){
            init_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_InitFn;
            exec_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_ExecFn;
        }else if(M == 2){
            init_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_InitFn;
            exec_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_ExecFn;
        }else if(M == 4){
            init_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_InitFn;
            exec_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_ExecFn;
        }else{
            init_fn = FQBMMA_6x6xtrue_8x16x256_48x48x128_8x8x128_4_1_InitFn;
            exec_fn = FQBMMA_6x6xtrue_8x16x256_48x48x128_8x8x128_4_1_ExecFn;
        }
        state = (*init_fn)(
            reinterpret_cast<const int*>(A), B, x_scale_half, w_scale_half, M, N, K, D, 128, bias);
    }
    // w6a8
    else if (w_bits_ == 6 && x_bits_ == 8) {
        if(M == 1){
            init_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_4_1_InitFn;
            exec_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_4_1_ExecFn;
        }else if(M == 2){
            init_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_InitFn;
            exec_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_ExecFn;
        }else if(M == 4){
            init_fn = FQBMMA_8x6xtrue_4x64x256_32x48x128_8x8x128_4_1_InitFn;
            exec_fn = FQBMMA_8x6xtrue_4x64x256_32x48x128_8x8x128_4_1_ExecFn;
        }else{
            init_fn = FQBMMA_8x6xtrue_8x64x384_64x48x128_8x8x128_2_1_InitFn;
            exec_fn = FQBMMA_8x6xtrue_8x64x384_64x48x128_8x8x128_2_1_ExecFn;
        }
        state = (*init_fn)(
            reinterpret_cast<const int*>(A), B, x_scale_half, w_scale_half, M, N, K, D, 128, bias);
    } else {
        printf("[FlexQ][Error] unsupport w%da%d\n", w_bits_, x_bits_);
        return;
    }

    if (!state.initSuccess) {
        printf("[FlexQ][Error] return due to unsuccessful initialization.\n");
        return;
    }
    (*exec_fn)(state, stream);
}

void FLEXQGEMMWrapper::gemm(const int    M,
                          const int    N,
                          const int    K,
                          const half*  A,
                          const int*   B,
                          const half*  C,
                          half*        D,
                          float* x_scale,
                          const float* w_scale,
                          const float* scale_inter,
                          const float* scale_out,
                          bool         bias,
                          char*        flexq_gemm_workspace,
                          size_t       flexq_gemm_ws_bytes,
                          cudaStream_t stream = NULL)
{   
    half *x_scale_half = reinterpret_cast<half*>(x_scale);
    const half *w_scale_half = reinterpret_cast<const half*>(w_scale);
    // quant+packing
    pack(A, reinterpret_cast<int*>(flexq_gemm_workspace), x_scale_half, M, K, x_bits_, stream);

    gemm(M, N, K, reinterpret_cast<const int*>(flexq_gemm_workspace), B, C, D, 
        x_scale, w_scale, scale_inter, scale_out, bias, flexq_gemm_workspace, flexq_gemm_ws_bytes, stream);
}


/*
***   kernel optimum config   ***

------------A6000-48GB------------
model: LLaMA-7B
1) W6A6
init_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_8x16x256_48x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_8x16x256_48x48x128_8x8x128_4_1_ExecFn;

2) W6A8
init_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_4_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_4x64x256_32x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_4x64x256_32x48x128_8x8x128_4_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_8x64x384_64x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_8x64x384_64x48x128_8x8x128_2_1_ExecFn;


model: LLaMA-13B
1) W6A6
init_fn = FQBMMA_6x6xtrue_1x32x512_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_1x32x512_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_2x32x512_16x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_4x32x512_24x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_2_1_ExecFn;

2) W6A8
init_fn = FQBMMA_8x6xtrue_1x32x512_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_1x32x512_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_2x32x256_16x48x128_8x8x128_4_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_4x32x384_32x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_4x32x384_32x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_8x32x384_64x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_8x32x384_64x48x128_8x8x128_2_1_ExecFn;

model: LLaMA-30B
1) W6A6
init_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_1x32x256_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_3_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_3_1_ExecFn;

2) W6A8
init_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_8x16x512_64x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_8x16x512_64x48x128_8x8x128_4_1_ExecFn;


model: OPT-30B
1) W6A6
init_fn = FQBMMA_6x6xtrue_1x16x256_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_1x16x256_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_3_1_InitFn;
exec_fn = FQBMMA_6x6xtrue_8x32x256_48x48x128_8x8x128_3_1_ExecFn;

2) W6A8
init_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_2_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_1x32x256_8x48x128_8x8x128_2_1_ExecFn;
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
init_fn = FQBMMA_8x6xtrue_8x16x512_64x48x128_8x8x128_4_1_InitFn;
exec_fn = FQBMMA_8x6xtrue_8x16x512_64x48x128_8x8x128_4_1_ExecFn;
*/