#pragma once

#include "common/base.h"
#include "bgemm.cuh"

template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // CTA 3D tiling layout - x-dim stride
    int CTA_TILE_STRIDE,
    // type of accumulator
    typename AccumulatorType,
    // // type of shared memory swizzling
    // typename ASwizzle, typename BSwizzle, typename CSwizzle,
    // pipeline configuration
    bool UseRegisterDoubleBuffer = true, bool UseMinimumSync = true, bool GridMappingXYToMN = false
    >
struct FQBMMAKernel {
    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
    static constexpr int BLOCK_M = ThreadBlockShape::M;
    static constexpr int BLOCK_N = ThreadBlockShape::N;
    static constexpr int BLOCK_K = ThreadBlockShape::K;
    static constexpr int WARP_M = WarpShape::M;
    static constexpr int WARP_N = WarpShape::N;
    static constexpr int WARP_K = WarpShape::K;
    static constexpr int MMA_M = MmaShape::M;
    static constexpr int MMA_N = MmaShape::N;
    static constexpr int MMA_K = MmaShape::K;
    static constexpr int SKEW = W_BITS * BLOCK_N % 16 == 0 ? 4 : 0;
    static constexpr bool quant_signed = QuantType::SIGNED;
    static constexpr int WARP_M_TILES = WARP_M / MMA_M;
    static constexpr int WARP_N_TILES = WARP_N / MMA_N;
    static constexpr int WARPS_M_NUMS = CEIL(BLOCK_M * X_BITS, MMA_M) / WARP_M_TILES;
    static constexpr int WARPS_N_NUMS = CEIL(BLOCK_N * W_BITS, MMA_N) / WARP_N_TILES;
    static constexpr int GROUP_SIZE = 128;

    static constexpr int BLOCK_K_TILES = BLOCK_K / MMA_K;  // B = log2(BLOCK_K_TILES)
    static constexpr int swizzle_config_B = log2_constexpr(BLOCK_K_TILES);
    static constexpr int MMA_K_VEC_LENS = 4;  // M = log2(4) = 2
    static constexpr int swizzle_config_M = 2;
    static constexpr int BLOCK_K_NUMS_PER_32BANKS = 32 / MMA_K_VEC_LENS / BLOCK_K_TILES;  // S = B + log2(BLOCK_K_NUMS_PER_32BANKS)
    static constexpr int swizzle_config_S = swizzle_config_B + log2_constexpr(BLOCK_K_NUMS_PER_32BANKS);
    using ASwizzleType = SwizzleIdentity;
    using BSwizzleType = SwizzleIdentity;

    static_assert(WARP_K == MMA_K, "Only support warp shape K == Mma shape K.\n");
    static_assert(WARP_M % MMA_M == 0, "WARP_M must be an integer multiple of MMA_M.\n");
    static_assert(WARP_N % MMA_N == 0, "WARP_N must be an integer multiple of MMA_N.\n");
    static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be an integer multiple of WARP_K.\n");
    static_assert(kThreadBlockStage > 1, "kThreadBlockStage must be greater than 1.\n");
    static_assert(WARP_K % 128 == 0, "Only support warp shape WARP_K>=128 for performance.\n");
    static_assert(GROUP_SIZE == 128, "Only support group size = 128.\n");
    static_assert(MMA_K == 128, "Only support Mma shape K = 128.\n");
    // precompute constants
    static constexpr bool GridMapping = GridMappingXYToMN;
    // determine the number of threads
    static constexpr int blockDims = 32 * WARPS_M_NUMS * WARPS_N_NUMS;
#if GPU_ARCH >= 80
    // use multi-stage shared-mem buffer (needed by async copy)
    // data is copied directly from globalmem to shared memory without going through registers.
    static constexpr size_t input_buffer_size_static =
        kThreadBlockStage * BLOCK_M * BLOCK_K * X_BITS / 8 +
        kThreadBlockStage * BLOCK_N * BLOCK_K * W_BITS / 8 + 
        kThreadBlockStage * SCALE_SIZE_X(BLOCK_M) * BLOCK_K / GROUP_SIZE * 4 + 
        kThreadBlockStage * SCALE_SIZE_W(BLOCK_N) * BLOCK_K / GROUP_SIZE * 4;
#else
    // Not supported
#endif // GPU_ARCH >= 80

    // The output results need to be stored in shem for scaling processing.
    // static constexpr size_t output_buffer_size_static =
    //     (MMA_M * WARP_M_TILES * WARPS_M_NUMS) * (MMA_N * WARP_N_TILES * WARPS_N_NUMS + SKEW) *
    //     sizeof(int32_t);
    static constexpr size_t output_buffer_size_static =
        BLOCK_M * BLOCK_N * sizeof(half);

    // mainloop interface
    __device__ __forceinline__ void mainLoop(const int M, const int N, const int K, const int *X,
                                             const int *W, const half *X_SCALE, const half *W_SCALE, half *D, int *shared_mem_workspace, int group_size, bool bias = false);
};

#if GPU_ARCH >= 80
// async-copy multi-stage kernel
// RowMajor * ColMajor(A:activation,  B:weight)
template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // CTA 3D tiling layout - x-dim stride
    int CTA_TILE_STRIDE,
    // type of accumulator
    typename AccumulatorType,
    // // type of shared memory swizzling
    // typename ASwizzle, typename BSwizzle, typename CSwizzle,
    // pipeline configuration
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN>
__device__ __forceinline__ void
FQBMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, kThreadBlockStage, CTA_TILE_STRIDE, AccumulatorType,
             UseRegisterDoubleBuffer, UseMinimumSync,
             GridMappingXYToMN>::mainLoop(const int M, const int N, const int K, const int *X,
                                          const int *W, const half *X_SCALE, const half *W_SCALE, half *D, int *shared_mem_workspace, int group_size, bool bias)
{
    // compute some global ids,
    const unsigned int idx_block_M = GridMappingXYToMN ? (blockIdx.z * gridDim.x + blockIdx.x) : ((blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : (blockIdx.y));
    const unsigned int idx_block_N = GridMappingXYToMN ? ((blockIdx.z % 2) ? (gridDim.y - blockIdx.y - 1) : (blockIdx.y)) : (blockIdx.z * gridDim.x + blockIdx.x);
    const unsigned int warp_id = threadIdx.x >> 5;

    const int tid = threadIdx.x & 31;
    const int tid_row = tid >> 2;
    const int tid_col = tid % 4;

    constexpr int chunk_m = BLOCK_M < MMA_M ? BLOCK_M : MMA_M;
    constexpr int chunk_n = BLOCK_N < MMA_N ? BLOCK_N : MMA_N;

    constexpr int chunk_m_num = BLOCK_M / chunk_m;
    constexpr int chunk_n_num = BLOCK_N / chunk_n;

    // compute global offsets: 0 Bit component position
    constexpr int x_chunk_m_row_offset = MMA_K / 32;
    constexpr int w_chunk_n_row_offset = MMA_K / 32;
    constexpr int x_bit_offset = chunk_m * x_chunk_m_row_offset;
    constexpr int w_bit_offset = chunk_n * w_chunk_n_row_offset;
    constexpr int x_chunk_m_offset = x_bit_offset * X_BITS;
    constexpr int w_chunk_n_offset = w_bit_offset * W_BITS;
    int x_k_tile_offset = x_chunk_m_offset * M / chunk_m;
    int w_k_tile_offset = w_chunk_n_offset * N / chunk_n;

    const int *x_panel = X + idx_block_M * BLOCK_M / chunk_m * x_chunk_m_offset;
    const int *w_panel = W + idx_block_N * BLOCK_N / chunk_n * w_chunk_n_offset;

    const int *x_scale_panel = (int *)(X_SCALE) + idx_block_M * SCALE_SIZE_X(BLOCK_M);
    const int *w_scale_panel = (int *)(W_SCALE) + idx_block_N * SCALE_SIZE_W(BLOCK_N);

    // compute shared memory buffer addresses
    constexpr int NStage = kThreadBlockStage;
    constexpr int size_of_tile_x = BLOCK_M * BLOCK_K * X_BITS; // Single-stage offset for 1-bit quantization
    constexpr int size_of_tile_w = BLOCK_N * BLOCK_K * W_BITS; // Single-stage offset for 1-bit quantization
    constexpr int size_of_tile_output = BLOCK_M * BLOCK_N; // half
    constexpr int size_of_tile_x_scale = SCALE_SIZE_X(BLOCK_M) * BLOCK_K / GROUP_SIZE; // int(half2)
    constexpr int size_of_tile_w_scale = SCALE_SIZE_W(BLOCK_N) * BLOCK_K / GROUP_SIZE; // int(half2)
    int *shared_x = shared_mem_workspace;
    int *shared_w = shared_x + size_of_tile_x * NStage / 32;
    int *shared_x_scale = shared_w + size_of_tile_w * NStage / 32;
    int *shared_w_scale = shared_x_scale + size_of_tile_x_scale * NStage;
    constexpr int k_tile_chunk = MMA_K / BITS_INT;
    constexpr int kAccess = 128;
    constexpr int iter_copy_x = CEIL(size_of_tile_x / kAccess, blockDims);
    constexpr int iter_copy_w = CEIL(size_of_tile_w / kAccess, blockDims);
    constexpr int outputAccess = 8;
    // constexpr int iter_copy_output = CEIL(size_of_tile_output / outputAccess, blockDims);
    constexpr int scale_kAccess = 4;  // 4 * half2 = 4 * 4 Bytes = 16 Bytes
    constexpr int iter_copy_x_scale = CEIL(size_of_tile_x_scale / scale_kAccess, blockDims);
    constexpr int iter_copy_w_scale = CEIL(size_of_tile_w_scale / scale_kAccess, blockDims);

    // Assume that N is divisible by CTA
    // bool is_residue =
    //     (M % BLOCK_M != 0) && (idx_block_M == (GridMappingXYToMN ? gridDim.x : gridDim.y) - 1);

    // shared_mem X: [kThreadBlockStage, BLOCK_K / 128, BLOCK_M / chunk_m, X_BITS, chunk_m, 4]
    // shared_mem W: [kThreadBlockStage, BLOCK_K / 128, BLOCK_N / chunk_n, W_BITS, chunk_n, 4]
    int x_row = warp_id / WARPS_N_NUMS * WARP_M;
    int w_row = warp_id % WARPS_N_NUMS * WARP_N;

    int x_row_nums = warp_id / WARPS_N_NUMS * MMA_M;
    int w_row_nums = warp_id % WARPS_N_NUMS * MMA_N;

    // smem X row major
    constexpr const int smem_chunk_m_ldx = MMA_K / 32;
    constexpr const int smem_bit_ldx = chunk_m * smem_chunk_m_ldx;
    constexpr const int smem_row_ldx = smem_bit_ldx * X_BITS;
    constexpr const int smem_k_tile_ldx = smem_row_ldx * BLOCK_M / chunk_m;
    // smem W col major [K, N] colmajor = [N, K] rowmajor
    constexpr const int smem_chunk_n_ldw = MMA_K / 32;
    constexpr const int smem_bit_ldw = chunk_n * smem_chunk_n_ldw;
    constexpr const int smem_row_ldw = smem_bit_ldw * W_BITS;
    constexpr const int smem_k_tile_ldw = smem_row_ldw * BLOCK_N / chunk_n;

    constexpr const int smem_ldx_scale = SCALE_SIZE_X(BLOCK_M);
    constexpr const int smem_ldw_scale = SCALE_SIZE_W(BLOCK_N);

    ASwizzleType aSwizzle;
    BSwizzleType bSwizzle;

    // define mma buffers
    typedef typename fq_bmma::fragment_a_rowmajor<MmaShape> FragmentA;
    typedef typename fq_bmma::fragment_b_colmajor<MmaShape> FragmentB;
    typedef typename fq_bmma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    typedef typename fq_bmma::fragment_c<MmaShape, float> FragmentC_FLOAT;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][WARP_M_TILES];
    FragmentB bfrag[kWarpStage][WARP_N_TILES];
    FragmentC cfrag[WARP_M_TILES][WARP_N_TILES];

    int32_t a_scale[kWarpStage][1] = {0};
    int32_t b_scale[kWarpStage][1] = {0};
    FragmentC_FLOAT cfrag_float[WARP_M_TILES][WARP_N_TILES];
    
    const int num_tile = CEIL(K, BLOCK_K);
    Pipeline<NStage, UseMinimumSync> pipe;
    int fetch = 0, compute = 0;

#pragma unroll
    for (; compute < num_tile; compute++) {
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquireWriter();
            // fetch data
            if (fetch < num_tile) {
                // current fetch stage`s src global mem: 0 bit position
                const int *tile_x = x_panel + fetch * (BLOCK_K / MMA_K) * M * X_BITS * 4;
                const int *tile_w = w_panel + fetch * (BLOCK_K / MMA_K) * N * W_BITS * 4;
                
                // current fetch stage`s dst shared_mem
                int *shared_tile_x = shared_x + (fetch % NStage) * size_of_tile_x / 32;
                int *shared_tile_w = shared_w + (fetch % NStage) * size_of_tile_w / 32;
                
#pragma unroll
                for (int i = 0; i < iter_copy_x; i++) {
                    // [BLOCK_K / 128, BLOCK_M / chunk_m, X_BITS, chunk_m, 4]
                    int idx = (threadIdx.x + blockDims * i) * kAccess / 32;
                    int idx_k_tile = idx / (4 * chunk_m * X_BITS * chunk_m_num);
                    int idx_chunk_m_id = idx / (4 * chunk_m * X_BITS) % chunk_m_num;
                    int idx_bit = idx / (4 * chunk_m) % X_BITS;
                    int idx_chunk_m_row = (idx / 4) % chunk_m;

                    bool valid = (idx < size_of_tile_x / 32);
                    // residue handling
                    bool zfill = ((fetch * BLOCK_K + idx_k_tile * 128) >= K) ||
                                 (idx_chunk_m_id * chunk_m + idx_chunk_m_row >= (M - idx_block_M * BLOCK_M)) || (idx_bit >= X_BITS);
                    // [idx_k, idx_chunk_m_id, idx_bit, idx_chunk_m_row] and The starting address of the current CTA block in the 0bit matrix is ​​tile_x
                    const int *src =
                        tile_x + idx_k_tile * x_k_tile_offset + idx_chunk_m_id * x_chunk_m_offset + idx_bit * x_bit_offset + idx_chunk_m_row * x_chunk_m_row_offset;
                    int *dst = shared_tile_x + aSwizzle(idx);
                    // Global mem loads data to shard mem, copy 128 b1 = int4 = 16 bytes
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }
#pragma unroll
                for (int i = 0; i < iter_copy_w; i++) {
                    // [BLOCK_K / 128, BLOCK_N / chunk_n, W_BITS, chunk_n, 4]
                    int idx = (threadIdx.x + blockDims * i) * kAccess / 32;
                    int idx_k_tile = idx / (4 * chunk_n * W_BITS * chunk_n_num);
                    int idx_chunk_n_id = idx / (4 * chunk_n * W_BITS) % chunk_n_num;
                    int idx_bit = idx / (4 * chunk_n) % W_BITS;
                    int idx_chunk_n_row = (idx / 4) % chunk_n;

                    bool valid = (idx < size_of_tile_w / 32);
                    bool zfill = ((fetch * BLOCK_K + idx_k_tile * 128) >= K) ||
                                 (idx_chunk_n_id * chunk_n + idx_chunk_n_row >= (N - idx_block_N * BLOCK_N)) || (idx_bit >= W_BITS);
                    // [idx_k, idx_chunk_n_id, idx_bit, idx_chunk_n_row] and The starting address of the current CTA block in the 0bit matrix is ​​tile_w
                    const int *src =
                        tile_w + idx_k_tile * w_k_tile_offset + idx_chunk_n_id * w_chunk_n_offset + idx_bit * w_bit_offset + idx_chunk_n_row * w_chunk_n_row_offset;
                    int *dst = shared_tile_w + bSwizzle(idx);
                    // Global mem loads data to shard mem, copy 128 b1 = int4 = 16 bytes
                    cpAsyncPredZfillCacheEvict<16>(dst, src, valid, zfill);
                }

                const int *tile_x_scale = x_scale_panel + fetch * BLOCK_K / GROUP_SIZE * SCALE_SIZE_X(M);
                const int *tile_w_scale = w_scale_panel + fetch * BLOCK_K / GROUP_SIZE * SCALE_SIZE_W(N);
                int *shared_tile_x_scale = shared_x_scale + (fetch % NStage) * size_of_tile_x_scale;
                int *shared_tile_w_scale = shared_w_scale + (fetch % NStage) * size_of_tile_w_scale;

#pragma unroll
                for(int i = 0; i < iter_copy_x_scale; i++){
                    // [BLOCK_K / GROUP_SIZE, SCALE_SIZE_X(BLOCK_M)]
                    int idx = (threadIdx.x + blockDims * i) * scale_kAccess;
                    int idx_k = idx / SCALE_SIZE_X(BLOCK_M);
                    int idx_m = idx % SCALE_SIZE_X(BLOCK_M);
                    bool valid = (idx < size_of_tile_x_scale);
                    // residue handling
                    bool zfill = ((fetch * BLOCK_K + idx_k * GROUP_SIZE) >= K) ||
                                 (idx_m >= (M - idx_block_M * BLOCK_M));
                    const int *src =
                        tile_x_scale + (idx_k * SCALE_SIZE_X(M) + idx_m); 
                    int *dst = shared_tile_x_scale + idx;
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }

#pragma unroll
                for(int i = 0; i < iter_copy_w_scale; i++){
                    // [BLOCK_K / GROUP_SIZE, SCALE_SIZE_W(BLOCK_N)]
                    int idx = (threadIdx.x + blockDims * i) * scale_kAccess;
                    int idx_k = idx / SCALE_SIZE_W(BLOCK_N);
                    int idx_n = idx % SCALE_SIZE_W(BLOCK_N);
                    bool valid = (idx < size_of_tile_w_scale);
                    bool zfill = ((fetch * BLOCK_K + idx_k * GROUP_SIZE) >= K) ||
                                 (idx_n >= (N - idx_block_N * BLOCK_N));
                    const int *src =
                        tile_w_scale + (idx_k * SCALE_SIZE_W(N) + idx_n);
                    int *dst = shared_tile_w_scale + idx;
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }
            }
            pipe.commitStage();
        }
        pipe.acquireReader();

        int *shared_tile_x = shared_x + (compute % NStage) * size_of_tile_x / 32;
        int *shared_tile_w = shared_w + (compute % NStage) * size_of_tile_w / 32;
        int *shared_tile_x_scale = shared_x_scale + (compute % NStage) * size_of_tile_x_scale;
        int *shared_tile_w_scale = shared_w_scale + (compute % NStage) * size_of_tile_w_scale;
        
#pragma unroll
        // compute [WARP_M_TILES, WARP_N_TILES]`s [MMA_M, MMA_N] in [chunk_m_id * X_BITS * chunk_m, chunk_n_id * W_BITS * chunk_n]
        // MMA_K == WARP_K Double_buffer is performed inside block
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            // Load into afrag
#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
                int offset = k * smem_k_tile_ldx + (x_row + m * MMA_M) * smem_chunk_m_ldx;
                fq_bmma::loadMatrixSync<ASwizzleType>(afrag[k % kWarpStage][m], shared_tile_x, offset, smem_chunk_m_ldx);
            }
            // Load into bfrag
#pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                int offset = k * smem_k_tile_ldw + (w_row + n * MMA_N) * smem_chunk_n_ldw;
                fq_bmma::loadMatrixSync<BSwizzleType>(bfrag[k % kWarpStage][n], shared_tile_w, offset, smem_chunk_n_ldw);
            }

            // Load scale into a_scale & b_scale
            int x_scale_offset = k * smem_ldx_scale + x_row_nums;
            const int *a_scale_src = shared_tile_x_scale + x_scale_offset + (tid_row % BLOCK_M);
            a_scale[k % kWarpStage][0] = *(int32_t *)a_scale_src;

            int w_scale_offset = k * smem_ldw_scale + SCALE_SIZE_W(w_row_nums);
            const int *b_scale_src = shared_tile_w_scale + w_scale_offset + (tid_col % SCALE_SIZE_W(BLOCK_N));
            b_scale[k % kWarpStage][0] = *(int32_t *)b_scale_src;

#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
                for (int n = 0; n < WARP_N_TILES; n++) {
                    fq_bmma::bmmaSync(cfrag[m][n], afrag[k % kWarpStage][m], bfrag[k % kWarpStage][n]);
                }
            }
            
            // dequant
            half2 row_scale = *(half2*)(&a_scale[k % kWarpStage][0]);
            half2 col_scale = *(half2*)(&b_scale[k % kWarpStage][0]);
            half2 rs_scale = __hmul2(row_scale, col_scale);
            float rs_scale_l = __half2float(rs_scale.x);  // a_scale[0] * b_scale[0]
            float rs_scale_r = __half2float(rs_scale.y);  // a_scale[0] * b_scale[1]

#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
                for (int n = 0; n < WARP_N_TILES; n++) {
                    cfrag_float[m][n].x[0] += cfrag[m][n].x[0] * rs_scale_l;
                    cfrag_float[m][n].x[1] += cfrag[m][n].x[1] * rs_scale_r;
                }
            }
        }
        pipe.releaseReader();
    }
    __syncthreads();

    int *shared_c = shared_mem_workspace;
    float *shared_c_float = (float *)shared_mem_workspace;
    half2 *shared_c_half2 = (half2 *)shared_mem_workspace;
    constexpr int smem_ldc = BLOCK_N / 2 + SKEW;
    int c_warp_offset = x_row_nums * smem_ldc + w_row_nums / 2;

    FragmentC_FLOAT bits_sum;
#pragma unroll
    for (int m = 0; m < WARP_M_TILES; m++) {
        int tid_x = m * MMA_M + tid_row;
        int tid_x_bit = tid_x / chunk_m % X_BITS;
#pragma unroll
        for (int n = 0; n < WARP_N_TILES; n++) {
            int tid_w = n * MMA_N + tid_col;
            int tid_w_bit = tid_w / chunk_n % W_BITS;

            int temp_mul = (1 << (tid_x_bit + tid_w_bit));
            temp_mul = ((tid_x_bit == X_BITS - 1) ^ (tid_w_bit == W_BITS - 1)) ? - temp_mul : temp_mul;

            if(tid_x / chunk_m >= X_BITS || tid_w / chunk_n >= W_BITS){
                cfrag_float[m][n].x[0] = cfrag_float[m][n].x[1] = 0.0f;
            }

            bits_sum.x[0] += temp_mul * cfrag_float[m][n].x[0];
            bits_sum.x[1] += temp_mul * cfrag_float[m][n].x[1];
        }
    }


/*
warp bit reduction
chunk_m = 1    reduction_turn = 3   shuffle_tid = 16 8 4
chunk_m = 2    reduction_turn = 2   shuffle_tid = 16 8
chunk_m = 4    reduction_turn = 1   shuffle_tid = 16
chunk_m = 8    reduction_turn = 0   shuffle_tid = 
*/
int shuffle_tid = WARP_SIZE / 2;
#pragma unroll
    for(int i = chunk_m; i < MMA_M; i*=2){
        float num1 = __shfl_down_sync(0xFFFFFFFF, bits_sum.x[0], shuffle_tid);
        float num2 = __shfl_down_sync(0xFFFFFFFF, bits_sum.x[1], shuffle_tid);
        // float num1 = __shfl_sync(0xFFFFFFFF, bits_sum.x[0], (tid + shuffle_tid) % WARP_SIZE, WARP_SIZE);
        // float num2 = __shfl_sync(0xFFFFFFFF, bits_sum.x[1], (tid + shuffle_tid) % WARP_SIZE, WARP_SIZE);
        bits_sum.x[0] += num1;
        bits_sum.x[1] += num2;

        shuffle_tid /= 2;
    }

    half2 bits_sum_half2 = __floats2half2_rn(bits_sum.x[0], bits_sum.x[1]);

    fq_bmma::storeMatrixSyncHalf2M8N8K128(bits_sum_half2, shared_c_half2, c_warp_offset, smem_ldc);
    __syncthreads();

    // Parallel reading and writing implementation
    int idx = threadIdx.x * outputAccess;
    int idx_m = idx / BLOCK_N;
    int idx_n = idx % BLOCK_N;
    bool valid = (idx < BLOCK_M * BLOCK_N && idx_m < M);
    if (valid){
        // half *shmem_stream_ptr = (half *)shared_mem_workspace + idx;
        half *shmem_stream_ptr = (half *)shared_mem_workspace + idx_m * smem_ldc * 2 + idx_n;
        int gmem_idx = idx_block_M * BLOCK_M * N + idx_block_N * BLOCK_N +
            idx_m * N + idx_n;
        int4 *s_ptr = reinterpret_cast<int4 *>(shmem_stream_ptr);
        int4 *d_ptr = reinterpret_cast<int4 *>(D + gmem_idx);
        *d_ptr = *s_ptr;
    }
}

#else // __CUDA_ARCH__ < 80
// Not supported
#endif
