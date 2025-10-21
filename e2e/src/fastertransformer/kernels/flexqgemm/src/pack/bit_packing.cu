#include "bit_packing.h"
#include <cuda_runtime.h>

#define WARP_SIZE 32

#define WARP_M 1
#define WARP_K 32

#define MMA_M 8
#define GROUP_SIZE 128

#define THREADS_NUM 128
#define WARP_PER_BLOCK (THREADS_NUM / WARP_SIZE) 

#define M_WARP_NUM 1
#define K_WARP_NUM 4
#define BLOCK_M (WARP_M * M_WARP_NUM)
#define BLOCK_K (WARP_K * K_WARP_NUM)

__forceinline__ __host__ __device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

// bit packing on quantized activations
__global__ void flexq_bit_packing_kernel(const int* __restrict__ T_in, int* bit_T_out,  
                    const int height, const int width, const int bitWidth)
{
    const unsigned laneid = threadIdx.x % WARP_SIZE;
    const unsigned warpid = threadIdx.x / WARP_SIZE;

    const int chunk_M = min(height, MMA_M);

    // shmem: [[BLOCK_M, BLOCK_K / BITS_INT](bit0), [BLOCK_M, BLOCK_K / BITS_INT](bit1), ...... , [BLOCK_M, BLOCK_K / BITS_INT](bitWidth)]
    extern __shared__ int shmem[];

    // BLOCK: [height / BLOCK_M, width / BLOCK_K]
    const int gdx = STEP_Y(height, BLOCK_M);
    const int gdy = STEP_Y(width, BLOCK_K);
    const int offset_row = STEP32(width);
    const int offset_bit = PAD_Y(height, BLOCK_M) * STEP_Y(width, BLOCK_K) * BLOCK_K / BITS_INT;
    const int offset_shmem_row = BLOCK_K / BITS_INT;
    const int offset_shmem_bit = BLOCK_M * BLOCK_K / BITS_INT;

    const int lx = warpid / K_WARP_NUM;
    const int ly = warpid % K_WARP_NUM;

    const int bx = blockIdx.x; // x index of the current block
    const int by = blockIdx.y; // y index of the current block

    for (int bitIdx = 0; bitIdx < bitWidth; bitIdx++){
        int f0 = ((bx * BLOCK_M + lx * WARP_M < height) && (by * BLOCK_K + ly * WARP_K + laneid < width)) ? \
            ((T_in[(bx * BLOCK_M + lx * WARP_M) * width + by * BLOCK_K + ly * WARP_K + laneid] >> bitIdx) & 1) : 0;

        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));

        if (laneid == 0){
            shmem[bitIdx * offset_shmem_bit + lx * WARP_M * offset_shmem_row + ly] = r0;
        }
    }

    __syncthreads();
    
    int output_lane_num = (bitWidth * BLOCK_M * BLOCK_K / BITS_INT4);
    for(int output_lane_id = threadIdx.x; output_lane_id < output_lane_num; output_lane_id += THREADS_NUM){
        const int output_bit = output_lane_id / (BLOCK_M * BLOCK_K / BITS_INT4);
        const int output_m = (output_lane_id % (BLOCK_M * BLOCK_K / BITS_INT4)) / (BLOCK_K / BITS_INT4);
        const int output_chunk_m_id = (bx * BLOCK_M + output_m) / chunk_M;
        const int output_chunk_m_row = (bx * BLOCK_M + output_m) % chunk_M;
        const int output_k = output_lane_id % (BLOCK_K / BITS_INT4);

        const int bit_T_out_index = (by * (BLOCK_K / BITS_INT4) + output_k) * (height * bitWidth * INT4_NUIT)
                                        + output_chunk_m_id * (bitWidth * chunk_M * INT4_NUIT) + output_bit * (chunk_M * INT4_NUIT) + output_chunk_m_row * INT4_NUIT;

        const int shmem_index = output_bit * offset_shmem_bit + output_m * BLOCK_K / BITS_INT + output_k * INT4_NUIT;
        
        *(reinterpret_cast<int4 *>(bit_T_out + bit_T_out_index)) = *((int4 *)(shmem + shmem_index));
    }

}

// Quantize and bit packing on unquantized activations.
__global__ void flexq_bit_packing_kernel(const half* __restrict__ T_in, int* bit_T_out, half* T_out_scale,
                    const int height, const int width, const int bitWidth)
{
    const unsigned laneid = threadIdx.x % WARP_SIZE;
    const unsigned warpid = threadIdx.x / WARP_SIZE;

    const int chunk_M = min(height, MMA_M);

    extern __shared__ int shmem[];

    const int gdx = STEP_Y(height, BLOCK_M);
    const int gdy = STEP_Y(width, BLOCK_K);
    const int offset_row = STEP32(width);
    const int offset_bit = PAD_Y(height, BLOCK_M) * STEP_Y(width, BLOCK_K) * BLOCK_K / BITS_INT;
    const int offset_shmem_row = BLOCK_K / BITS_INT;
    const int offset_shmem_bit = BLOCK_M * BLOCK_K / BITS_INT;

    const int lx = warpid / K_WARP_NUM;
    const int ly = warpid % K_WARP_NUM;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // A group of elements assigned to a single warp for processing.
    int threads_each_group = WARP_SIZE;
    // The number of warps required to process a single block of data.
    int need_warp_nums = BLOCK_M * BLOCK_K / GROUP_SIZE; 
    // The number of data elements processed per thread.
    int elements_each_thread = GROUP_SIZE / WARP_SIZE;
    // The index of the data group currently being processed by the thread.
    int group_id = threadIdx.x / threads_each_group;
    // The number of groups contained in the width (K).
    int row_group_nums = width / GROUP_SIZE;
    // The number of groups contained in the BLOCK_K.
    int BK_group_nums = BLOCK_K / GROUP_SIZE; 
    // Warp layout within a single block.
    int warp_group_row = group_id / BK_group_nums;
    int warp_group_col = group_id % BK_group_nums;
    
    if(warpid < need_warp_nums){
        int q_tid = threadIdx.x * elements_each_thread;
        int q_in_col = q_tid % BLOCK_K;
        int q_in_row = q_tid / BLOCK_K;
        const half2 *T_in_half2 = reinterpret_cast<const half2*>(T_in + (bx * BLOCK_M + q_in_row) * width + by * BLOCK_K + q_in_col);

        half input_frag[8];
        half maxv_h = -1;
#pragma unroll
        for(int i=0;i<elements_each_thread/2;i++){
            half2 temp = T_in_half2[i];
            input_frag[2*i] = temp.x;
            input_frag[2*i+1] = temp.y;
            maxv_h = __hmax(maxv_h, __habs(input_frag[2*i]));
            maxv_h = __hmax(maxv_h, __habs(input_frag[2*i+1]));
        }
        float maxv = __half2float(maxv_h);

        
#pragma unroll
        for(int i = threads_each_group / 2; i > 0; i >>= 1){
            float tmp;
            asm volatile(
                "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : 
                "=f"(tmp) : "f"(maxv), "r"(i)
            );
            maxv = (tmp > maxv ? tmp : maxv);
        }

        int lower_bound = -(1 << (bitWidth - 1));       // -2 ^ (bitWidth - 1)
        int upper_bound = (1 << (bitWidth - 1)) - 1;    // 2 ^ (bitWidth - 1) - 1
        maxv /= upper_bound;

        int row_offset = SCALE_PACKING_A(SCALE_SIZE_X(height));
        int scale_offset = (by * BK_group_nums + warp_group_col) * row_offset + SCALE_PACKING_A(bx * BLOCK_M + warp_group_row);
        if(threadIdx.x % threads_each_group == 0){
            T_out_scale[scale_offset] = T_out_scale[scale_offset + 1] = maxv;
        }

        float r_scale = __half2float(maxv);
        for(int i=0;i<elements_each_thread;i++){
            int val = (int)clamp(round(__half2float(input_frag[i]) / r_scale),
                                  lower_bound, upper_bound);
            
            shmem[threadIdx.x * elements_each_thread + i] = val;
        }

    }

    __syncthreads();

    int target_val = shmem[lx * WARP_M * WARP_K + ly * WARP_K + laneid];
    for (int bitIdx = 0; bitIdx < bitWidth; bitIdx++){
        int f0 = ((bx * BLOCK_M + lx * WARP_M < height) && (by * BLOCK_K + ly * WARP_K + laneid < width)) ? \
            ((target_val >> bitIdx) & 1) : 0;

        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));

        if (laneid == 0){
            shmem[bitIdx * offset_shmem_bit + lx * WARP_M * offset_shmem_row + ly] = r0;
        }

    }

    __syncthreads();

    int output_lane_num = (bitWidth * BLOCK_M * BLOCK_K / BITS_INT4);
    for(int output_lane_id = threadIdx.x; output_lane_id < output_lane_num; output_lane_id += THREADS_NUM){
        const int output_bit = output_lane_id / (BLOCK_M * BLOCK_K / BITS_INT4);
        const int output_m = (output_lane_id % (BLOCK_M * BLOCK_K / BITS_INT4)) / (BLOCK_K / BITS_INT4);
        const int output_chunk_m_id = (bx * BLOCK_M + output_m) / chunk_M;
        const int output_chunk_m_row = (bx * BLOCK_M + output_m) % chunk_M;
        const int output_k = output_lane_id % (BLOCK_K / BITS_INT4);

        const int bit_T_out_index = (by * (BLOCK_K / BITS_INT4) + output_k) * (height * bitWidth * INT4_NUIT)
                                        + output_chunk_m_id * (bitWidth * chunk_M * INT4_NUIT) + output_bit * (chunk_M * INT4_NUIT) + output_chunk_m_row * INT4_NUIT;

        const int shmem_index = output_bit * offset_shmem_bit + output_m * BLOCK_K / BITS_INT + output_k * INT4_NUIT;
        *(reinterpret_cast<int4 *>(bit_T_out + bit_T_out_index)) = *((int4 *)(shmem + shmem_index));
    }
}

void flexq_bit_packing(const int* in_data, int* packed_data, const int M, const int K, const int BIT, cudaStream_t stream){   
    dim3 threads(THREADS_NUM);
    dim3 blocks((M + BLOCK_M - 1) / BLOCK_M, (K + BLOCK_K - 1) / BLOCK_K);
    const size_t shmem_size = (BLOCK_M * BLOCK_K / BITS_INT * BIT) * sizeof(int);
    flexq_bit_packing_kernel<<<blocks, threads, shmem_size, stream>>>( \
        in_data, packed_data, \
        M, K, BIT);

    return ;
}

void flexq_bit_packing(const half* in_data, int* packed_data, half* T_out_scale, const int M, const int K, const int BIT, cudaStream_t stream){   
    dim3 threads(THREADS_NUM);
    dim3 blocks((M + BLOCK_M - 1) / BLOCK_M, (K + BLOCK_K - 1) / BLOCK_K);
    const size_t shmem_size = (BLOCK_M * BLOCK_K) * sizeof(int);
    flexq_bit_packing_kernel<<<blocks, threads, shmem_size, stream>>>( \
        in_data, packed_data, T_out_scale, \
        M, K, BIT);

    return ;
}