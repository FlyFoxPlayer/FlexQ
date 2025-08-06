#include "bit_packing.h"
#include <cuda_runtime.h>

#define WARP_SIZE 32

#define WARP_M 1
#define WARP_K 32

#define MMA_M 8

#define THREADS_NUM 128
#define WARP_PER_BLOCK (THREADS_NUM / WARP_SIZE) 

#define M_WARP_NUM 1
#define K_WARP_NUM 4
#define BLOCK_M (WARP_M * M_WARP_NUM)
#define BLOCK_K (WARP_K * K_WARP_NUM)

// ABQ-LLM bit packing kernel
__global__ void abq_packing_kernel(const uint4 *in_data, unsigned int *pack_data, const int m, const int k)
{
    const unsigned int bit = blockIdx.y;
    const int L = m * (k / 32);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L; idx += blockDim.x) {
        unsigned int pack_val = 0;
        // each threads read thirty two 32-bit elements to pack one 32-bit element
        // read 16B
        for (int i = 0; i < 8; ++i) {
            const uint4 val = in_data[idx * 8 + i];
            pack_val |= ((val.x >> bit) & 0x1) << (32 - 1 - i * 4);
            pack_val |= ((val.y >> bit) & 0x1) << (32 - 2 - i * 4);
            pack_val |= ((val.z >> bit) & 0x1) << (32 - 3 - i * 4);
            pack_val |= ((val.w >> bit) & 0x1) << (32 - 4 - i * 4);
        }
        //printf("m = %d, k = %d, L = %d, bit = %d, pack_val = %x\n", m, k, L, bit, pack_val);
        pack_data[bit * L + idx] = pack_val;
    }
}


// FlexQ bit packing kernel, --> [k / 128, M / chunk_M, x_bits, chunk_M, 4], chunk_M = min(chunk_M, MMA_M)
__global__ void flexq_packing_kernel(const int* __restrict__ T_in, int* bit_T_out,  
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

    // iterate through all bits
    for (int bitIdx = 0; bitIdx < bitWidth; bitIdx++){
        // boundry check whether inside, otherwise set to 0
        int f0 = ((bx * BLOCK_M + lx * WARP_M < height) && (by * BLOCK_K + ly * WARP_K + laneid < width)) ? \
            ((T_in[(bx * BLOCK_M + lx * WARP_M) * width + by * BLOCK_K + ly * WARP_K + laneid] >> bitIdx) & 1) : 0;

        // compressed, any thing outside boundry would be set to 0.
        // note that * f0 > 0 * in the 0/1 case. but >= 0 in 1/-1 case
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


void abq_bit_packing(const int *in_data, int *pack_data, int m, int k, int BIT,
                        cudaStream_t stream)
{
    dim3 threads(min(max(32, m * (k / 32)), 512));
    dim3 blocks((m * (k / 32) + threads.x - 1) / threads.x, BIT);
    abq_packing_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4 *>(in_data),
                                         reinterpret_cast<unsigned int *>(pack_data), m, k);
}



cudaError_t flexq_bit_packing(const int* in_data, int* packed_data, const int M, const int K, const int BIT, cudaStream_t stream){   
    dim3 threads(THREADS_NUM);
    dim3 blocks((M + BLOCK_M - 1) / BLOCK_M, (K + BLOCK_K - 1) / BLOCK_K);
    const size_t shmem_size = (BLOCK_M * BLOCK_K / BITS_INT * BIT) * sizeof(int);
    flexq_packing_kernel<<<blocks, threads, shmem_size, stream>>>( \
        in_data, packed_data, \
        M, K, BIT);
    cudaError_t ret = cudaGetLastError();
    return ret;
}

