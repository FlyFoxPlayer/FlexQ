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
#include "src/fastertransformer/kernels/flexqgemm/common/base.h"
#include "flexq_bmma_kernel.h"

struct FQBMMAOpState {
    size_t shared_mem_size;
    dim3 gridDim;
    dim3 blockDim;
    bool initSuccess = false;
    struct Argument_t {
        int M, N, K;
        const int *X;
        const int *W;
        half *X_SCALE;
        const half *W_SCALE;
        half *D;
        int group_size;
        bool bias = false;
    } args;
};

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage,
    // CTA 3D tiling layout
    int CTA_TILE_STRIDE
    >
class FQBMMAOp {
    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
    static constexpr bool SIGNED = QuantType::SIGNED;
    using AccumulatorType = int32_t;
public:
    FQBMMAOpState state;
    using KernelImpl =
        FQBMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, CTA_TILE_STRIDE, AccumulatorType, true, true, false>;
    void initialize(const int *X, const int *W, half* X_SCALE, const half* W_SCALE, int M, int N, int K, half *D, int group_size, bool bias);
    void operator()(cudaStream_t stream = NULL);
};

// *** device kernel ***
template <typename KernelImpl>
__global__ void launchFQBMMAKernel(typename FQBMMAOpState::Argument_t args)
{
    extern __shared__ int shared_mem_workspace[];
    KernelImpl k;
    k.mainLoop(args.M, args.N, args.K, args.X, args.W, args.X_SCALE, args.W_SCALE, args.D, shared_mem_workspace, args.group_size, args.bias);
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage,
    // CTA 3D tiling layout
    int CTA_TILE_STRIDE>
void FQBMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, CTA_TILE_STRIDE>::initialize(
    const int *X, const int *W, half* X_SCALE, const half* W_SCALE, int M, int N, int K, half *D, int group_size, bool bias)
{
    bool initSuccessFlag = true;

    // set argument
    this->state.args = FQBMMAOpState::Argument_t({ M, N, K, X, W, X_SCALE, W_SCALE, D, group_size, bias });
    // compute shared memory buffer size
    size_t input_buffer_size_dyn = 0;
    size_t input_buffer_size = input_buffer_size_dyn + KernelImpl::input_buffer_size_static;
    size_t output_buffer_size_dyn = 0;
    size_t output_buffer_size = output_buffer_size_dyn + KernelImpl::output_buffer_size_static;
    this->state.shared_mem_size = max(input_buffer_size, output_buffer_size);
    if (this->state.shared_mem_size >= 32 * 1024) {
        // set kernel attribute
        if (cudaSuccess != cudaFuncSetAttribute(launchFQBMMAKernel<KernelImpl>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                this->state.shared_mem_size) ||
            cudaSuccess != cudaFuncSetAttribute(launchFQBMMAKernel<KernelImpl>,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                100)) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err);
            // this->state.initSuccess = false;
            initSuccessFlag = false;
        }
    }

    // calculate launch configuration
    int gdimX = CTA_TILE_STRIDE;
    int gdimY =
        KernelImpl::GridMapping ? (CEIL(N, KernelImpl::BLOCK_N)) : (CEIL(M, KernelImpl::BLOCK_M));
    int gdimZ = 
        KernelImpl::GridMapping ? (CEIL(M, KernelImpl::BLOCK_M * CTA_TILE_STRIDE)) : (CEIL(N, KernelImpl::BLOCK_N * CTA_TILE_STRIDE));

    if(KernelImpl::GridMapping && (CEIL(M, KernelImpl::BLOCK_M) % CTA_TILE_STRIDE != 0 || CTA_TILE_STRIDE > CEIL(M, KernelImpl::BLOCK_M))){
        std::cerr << "Set kernel attribute failed: Illegal CTA_TILE-STRIDE parameter setting!\n";
        // this->state.initSuccess = false;
        initSuccessFlag = false;
    }

    if(!KernelImpl::GridMapping && (CEIL(N, KernelImpl::BLOCK_N) % CTA_TILE_STRIDE != 0 || CTA_TILE_STRIDE > CEIL(N, KernelImpl::BLOCK_N))){
        std::cerr << "Set kernel attribute failed: Illegal CTA_TILE-STRIDE parameter setting!\n";
        // this->state.initSuccess = false;
        initSuccessFlag = false;
    }
    
    this->state.gridDim = dim3(gdimX, gdimY, gdimZ);
    this->state.blockDim = dim3(KernelImpl::blockDims, 1, 1);

    if(initSuccessFlag)
        this->state.initSuccess = true;
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage,
    // CTA 3D tiling layout
    int CTA_TILE_STRIDE>
void FQBMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, CTA_TILE_STRIDE>::operator()(
    cudaStream_t stream)
{
    launchFQBMMAKernel<KernelImpl>
        <<<this->state.gridDim, this->state.blockDim, this->state.shared_mem_size, stream>>>(
            this->state.args);
}

// pure-function version of the original c++-object Op
// function handle easy for benchmarking, testing
template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage,
    // CTA 3D tiling layout
    int CTA_TILE_STRIDE>
FQBMMAOpState FQBMMAInitFn(const int *X, const int *W, half* X_SCALE, const half* W_SCALE, int M, int N, int K, half *D, int group_size, bool bias)
{
    FQBMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, CTA_TILE_STRIDE> op;
    op.initialize(X, W, X_SCALE, W_SCALE, M, N, K, D, group_size, bias);
    return op.state;
}

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // pipeline config
    int NStage,
    // CTA 3D tiling layout
    int CTA_TILE_STRIDE>
void FQBMMAExecFn(FQBMMAOpState &state, cudaStream_t stream = NULL)
{
    using KernelImpl =
        typename FQBMMAOp<QuantType, ThreadBlockShape, WarpShape, MmaShape, NStage, CTA_TILE_STRIDE>::KernelImpl;
    launchFQBMMAKernel<KernelImpl>
        <<<state.gridDim, state.blockDim, state.shared_mem_size, stream>>>(state.args);
}

typedef FQBMMAOpState (*FQBMMAInitFn_t)(const int *, const int *, half *, const half *, int, int, int, half *, int, bool);
typedef void (*FQBMMAExecFn_t)(FQBMMAOpState &, cudaStream_t);
