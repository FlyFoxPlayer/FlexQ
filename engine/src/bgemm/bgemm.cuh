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

namespace bmma_m8n8k128b1{
    /*
        link: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k128
        mma.m8n8k128
    */
    typedef struct fragment_a_row_major {
        uint32_t x[1];
        static const int num_elements = 1;
    } FragmentA;
    typedef struct fragment_b_col_major {
        uint32_t x[1];
        static const int num_elements = 1;
    } FragmentB;
    typedef struct fragment_c {
        int32_t x[2] = { 0 };
        static const int num_elements = 2;
    } FragmentC;
    typedef struct fragment_c_float {
        float x[2] = { 0.0f };
        static const int num_elements = 2;
    } FragmentC_FLOAT;

    template<typename Frag>
    __device__ __forceinline__ void reset_fragment(Frag &frag) {
        #pragma unroll
        for (int i = 0; i < Frag::num_elements; i++) {
            frag.x[i] = 0;
        }
    }
}

// Convert shared memory address to unsigned
__device__ __forceinline__ unsigned getShmemPtr(const void *ptr){
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// Load data from global memory to shared memory (async)
__device__ __forceinline__ void load_matrix_16B_async_g2s(void *shmem_ptr, void const *gmem_ptr, const bool valid = true){
    unsigned shmem_addr = getShmemPtr(shmem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global [%1], [%2], 16, 16;\n"
        // "  @p cp.async.ca.shared.global [%1], [%2], 16, 16;\n"
        "}\n" ::"r"((int)valid),
        "r"(shmem_addr), "l"(gmem_ptr)
    );
}

// Load data from global memory to shared memory (sync)
__device__ __forceinline__ void load_matrix_16B_sync_g2s(void *shmem_ptr, const void *gmem_ptr, const bool valid = true) {
    if (valid) {
        const uint4 data = __ldg(reinterpret_cast<const uint4*>(gmem_ptr));
        *reinterpret_cast<uint4*>(shmem_ptr) = data;
    }
}

// Load A (activation) matrix from shared memory to registers
__device__ __forceinline__ void load_A_matrix_sync_s2r(bmma_m8n8k128b1::FragmentA &a, const int *src, const int offset, const int ldm){
    const int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = lane_id % 4;
    const int *tar_src = src + offset + row * ldm + col;
    a.x[0] = *(uint32_t *)tar_src;
}

// Load A (activation) matrix from shared memory to registers (ldmatrix)
__device__ __forceinline__ void load_A_matrix_sync_s2r_ldmatrix_m8n8_x1_b16(bmma_m8n8k128b1::FragmentA &a, const int *src, const int offset, const int ldm){
    const int lane_id = threadIdx.x & 31;
    int row = lane_id;
    int col = 0;

    uint32_t* frag_a = reinterpret_cast<uint32_t*>(&a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(src + offset + row * ldm + col));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(frag_a[0]) : "r"(smem)
    );
}

// Load B (weight) matrix from shared memory to registers
__device__ __forceinline__ void load_B_matrix_sync_s2r(bmma_m8n8k128b1::FragmentB &b, const int *src, const int offset, const int ldm){
    const int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = lane_id % 4;
    const int *tar_src = src + offset + row * ldm + col;
    b.x[0] = *(uint32_t *)tar_src;
}

// mma m8n8k128 binary operation
__device__ __forceinline__ void bmma_m8n8k128b1_sync(bmma_m8n8k128b1::FragmentC &d,
                            const bmma_m8n8k128b1::FragmentA &a,
                            const bmma_m8n8k128b1::FragmentB &b,
                            const bmma_m8n8k128b1::FragmentC &c){
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
             "{%0, %1}, {%2}, {%3}, {%4,%5};\n"
             : "=r"(d.x[0]), "=r"(d.x[1])
             : "r"(a.x[0]), "r"(b.x[0]), "r"(c.x[0]), "r"(c.x[1]));
}

// C -- Zero
__device__ __forceinline__ void bmma_m8n8k128b1_c0_sync(bmma_m8n8k128b1::FragmentC &d,
                            const bmma_m8n8k128b1::FragmentA &a,
                            const bmma_m8n8k128b1::FragmentB &b){
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
             "{%0, %1}, {%2}, {%3}, {%4,%5};\n"
             : "=r"(d.x[0]), "=r"(d.x[1])
             : "r"(a.x[0]), "r"(b.x[0]), "r"(0), "r"(0));
}

// Store computed C (output) matrix from registers to shared memory
__device__ __forceinline__ void store_C_matrix_sync_r2s(const bmma_m8n8k128b1::FragmentC &c, int *dst, const int offset, const int ldm){
    int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = (lane_id % 4) * 2;
    int offset_ = offset + row * ldm + col;
    *(dst + offset_) = c.x[0];
    *(dst + offset_ + 1) = c.x[1];
}

// Load A (activation) matrix from global memory to registers
__device__ __forceinline__ void load_A_matrix_sync_g2r(
    bmma_m8n8k128b1::FragmentA &a, const int *src, const int offset, const int ldm){
    const int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = lane_id % 4;
    const int *tar_src = src + offset + row * ldm + col;
    a.x[0] = *(uint32_t *)tar_src;
}

// Load B (weight) matrix from global memory to registers
__device__ __forceinline__ void load_B_matrix_sync_g2r(
    bmma_m8n8k128b1::FragmentB &b, const int *src, const int offset, const int ldm){
    const int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = lane_id % 4;
    const int *tar_src = src + offset + row * ldm + col;
    b.x[0] = *(uint32_t *)tar_src;
}

// Write C matrix from registers to shared memory
__device__ __forceinline__ void store_C_matrix_sync_g2s(
    const bmma_m8n8k128b1::FragmentC &c, int *dst, const int offset, const int ldm){
    int lane_id = threadIdx.x & 31;
    int row = lane_id >> 2;
    int col = (lane_id % 4) * 2;
    int offset_ = offset + row * ldm + col;
    *(dst + offset_) = c.x[0];
    *(dst + offset_ + 1) = c.x[1];
}

// device function to convert shared memory address into unsigned format
DEVICE_INLINE unsigned getSmemPtr(const void *ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

DEVICE_INLINE
void copyAndSync(unsigned *dst, const unsigned *src, int size)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
    __syncthreads();
}

// 从全局内存加载数据至共享内存（异步）（缓存驱逐策略）
template <int SizeInBytes>
__device__ inline void cpAsyncPredZfillCacheEvict(void* smem_ptr, const void* gmem_ptr,
                                                    const bool pred_guard = true, const bool zfill = false) {
  unsigned smem_int_ptr = getSmemPtr(smem_ptr);
  int src_in_bytes = (zfill ? 0 : SizeInBytes);
  ASSEMBLY(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   .reg .b64 q;\n"
    "   @p createpolicy.fractional.L2::evict_first.b64 q, 1.0;\n"
    "   @p cp.async.cg.shared.global.L2::cache_hint [%1], [%2], %3, %4, q;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
  );
}


template <int SizeInBytes>
DEVICE_INLINE void cpAsyncPredZfill(void *smem_ptr, void const *gmem_ptr,
                                    const bool pred_guard = true, const bool zfill = false)
{
    unsigned smem_int_ptr = getSmemPtr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);
    ASSEMBLY("{\n"
             "  .reg .pred p;\n"
             "  setp.ne.b32 p, %0, 0;\n"
             "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
             "}\n" ::"r"((int)pred_guard),
             "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
}

template <int SizeInBytes>
DEVICE_INLINE void cpSyncPredZfill(void *shmem_ptr, const void *gmem_ptr,
                                    const bool pred_guard = true, const bool zfill = false) {
    if (pred_guard) {
        if(zfill){
            uint4 data;
            data.x = data.y = data.z = data.w = 0;
            *reinterpret_cast<uint4*>(shmem_ptr) = data;
        }else{
            const uint4 data = __ldg(reinterpret_cast<const uint4*>(gmem_ptr));
            *reinterpret_cast<uint4*>(shmem_ptr) = data;
        }

    }
}

namespace fq_bmma
{
    #if GPU_ARCH >= 80

    template <typename Shape> struct fragment_a_rowmajor;
    template <typename Shape> struct fragment_b_colmajor;
    template <typename Shape, typename Accumulator> struct fragment_c;
    template <bool trans, int num_reg, int nbit>
    DEVICE_INLINE void ldmatrix(uint32_t *dst, const void *src);

    // *** BMMA: 8x8x128 int32.b1 ***
    template <> struct fragment_a_rowmajor<ShapeBase<8, 8, 128>> {
        uint32_t x;
    };
    template <> struct fragment_b_colmajor<ShapeBase<8, 8, 128>> {
        uint32_t x;
    };
    template <> struct fragment_c<ShapeBase<8, 8, 128>, int32_t> {
        int32_t x[2] = { 0 };
    };
    template <> struct fragment_c<ShapeBase<8, 8, 128>, float> {
        float x[2] = { 0.0f };
    };
    template <class F>
    DEVICE_INLINE void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base,
                                    const int offset, const int ldm);
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base, const int offset,
                        const int ldm);
    template <class F>
    DEVICE_INLINE void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base,
                                    const int offset, const int ldm);
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base, const int offset,
                        const int ldm);
    DEVICE_INLINE
    void bmmaSync(fragment_c<ShapeBase<8, 8, 128>, int32_t> &d,
                const fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a,
                const fragment_b_colmajor<ShapeBase<8, 8, 128>> &b,
                const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c);
    template <class F>
    DEVICE_INLINE void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                                    const int offset, const int ldm);
    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                        const int offset, const int ldm);

    DEVICE_INLINE
    void storeMatrixSyncHalf2M8N8K128(const half2 c, half2 *base, const int offset, const int ldm);


    // *** BMMA: 16x8x128 int32.b1 ***
    template <> struct fragment_a_rowmajor<ShapeBase<16, 8, 128>> {
        uint32_t x[2];
    };
    template <> struct fragment_b_colmajor<ShapeBase<16, 8, 128>> {
        uint32_t x;
    };
    template <> struct fragment_c<ShapeBase<16, 8, 128>, int32_t> {
        int32_t x[4] = { 0 };
    };
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<16, 8, 128>> &a, const int *base,
                        const int offset, const int ldm);
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<16, 8, 128>> &b, const int *base,
                        const int offset, const int ldm);
    DEVICE_INLINE
    void bmmaSync(fragment_c<ShapeBase<16, 8, 128>, int32_t> &d,
                const fragment_a_rowmajor<ShapeBase<16, 8, 128>> &a,
                const fragment_b_colmajor<ShapeBase<16, 8, 128>> &b,
                const fragment_c<ShapeBase<16, 8, 128>, int32_t> &c);
    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<16, 8, 128>, int32_t> &c, int *base,
                        const int offset, const int ldm);

    // *** BMMA: 16x8x256 int32.b1 ***
    template <> struct fragment_a_rowmajor<ShapeBase<16, 8, 256>> {
        uint32_t x[4];
    };
    template <> struct fragment_b_colmajor<ShapeBase<16, 8, 256>> {
        uint32_t x[2];
    };
    template <> struct fragment_c<ShapeBase<16, 8, 256>, int32_t> {
        int32_t x[4] = { 0 };
    };
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<16, 8, 256>> &a, const int *base,
                        const int offset, const int ldm);
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<16, 8, 256>> &b, const int *base,
                        const int offset, const int ldm);
    DEVICE_INLINE
    void bmmaSync(fragment_c<ShapeBase<16, 8, 256>, int32_t> &d,
                const fragment_a_rowmajor<ShapeBase<16, 8, 256>> &a,
                const fragment_b_colmajor<ShapeBase<16, 8, 256>> &b,
                const fragment_c<ShapeBase<16, 8, 256>, int32_t> &c);
    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<16, 8, 256>, int32_t> &c, int *base,
                        const int offset, const int ldm);

    // *** implementation ***

    // ldmatrix
    // Unfortunately ldmatrix currently does not support the b1 data type
    template <bool trans, int num_reg, int nbit>
    DEVICE_INLINE void ldmatrix(uint32_t *dst, const void *src)
    {
        // no f32 transpose is supported in current cuda
        // static_assert((!trans) || nbit==16);
        unsigned smem_ptr = getSmemPtr(src);
        uint32_t *x = dst;
        if (!trans) {
            if (num_reg == 4) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                        : "r"(smem_ptr));
            } else if (num_reg == 2) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(x[0]), "=r"(x[1])
                        : "r"(smem_ptr));
            } else if (num_reg == 1) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(x[0])
                        : "r"(smem_ptr));
            } else
                assert(0);
        } else { // trans
            if (num_reg == 4) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                        : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                        : "r"(smem_ptr));
            } else if (num_reg == 2) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                        : "=r"(x[0]), "=r"(x[1])
                        : "r"(smem_ptr));
            } else if (num_reg == 1) {
                ASSEMBLY("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(x[0])
                        : "r"(smem_ptr));
            } else
                assert(0);
        }
    }

    // load a matrix [8, 128] rowmajor
    // ldm counts with integer pointers
    template <class F>
    DEVICE_INLINE void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base,
                                    const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4;
        F f;
        const int *src = base + f(offset + row * ldm + col);
        a.x = *(uint32_t *)src;
    }
    // ldm counts with integer pointers
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base, const int offset,
                        const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src = base + offset + row * ldm + col;
        a.x = *(uint32_t *)src;
    }

    // load b matrix [128, 8] colmajor = [8, 128] rowmajor
    // ldm counts with integer pointers
    // base data [mma_N, mma_K] rowmajor = [mma_K, mma_N] colmajor
    // So just follow the normal rowmajor thread allocation to read.
    template <class F>
    DEVICE_INLINE void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base,
                                    const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4;
        F f;
        const int *src = base + f(offset + row * ldm + col);
        b.x = *(uint32_t *)src;
    }
    // ldm counts with integer pointers
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base, const int offset,
                        const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src = base + offset + row * ldm + col;
        b.x = *(uint32_t *)src;
    }

    // a matrix [8, 128] rowmajor *  b matrix [128, 8] colmajor |  b matrix [8, 128] rowmajor
    DEVICE_INLINE void bmmaSync(fragment_c<ShapeBase<8, 8, 128>, int32_t> &d,
                                const fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a,
                                const fragment_b_colmajor<ShapeBase<8, 8, 128>> &b,
                                const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c)
    {
        ASSEMBLY("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
                "{%0, %1}, {%2}, {%3}, {%4,%5};\n"
                : "=r"(d.x[0]), "=r"(d.x[1])
                : "r"(a.x), "r"(b.x), "r"(c.x[0]), "r"(c.x[1]));
    }

    DEVICE_INLINE void bmmaSync(fragment_c<ShapeBase<8, 8, 128>, int32_t> &d,
                                const fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a,
                                const fragment_b_colmajor<ShapeBase<8, 8, 128>> &b)
    {
        ASSEMBLY("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
                "{%0, %1}, {%2}, {%3}, {%4,%5};\n"
                : "=r"(d.x[0]), "=r"(d.x[1])
                : "r"(a.x), "r"(b.x), "r"(0), "r"(0));
    }

    template <class F>
    DEVICE_INLINE void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                                    const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        F f;
        *(base + f(offset_)) = c.x[0];
        *(base + f(offset_ + 1)) = c.x[1];
    }

    template <class F>
    DEVICE_INLINE void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, float> &c, float *base,
                                    const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        F f;
        *(base + f(offset_)) = c.x[0];
        *(base + f(offset_ + 1)) = c.x[1];
    }

    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        *(base + offset_) = c.x[0];
        *(base + offset_ + 1) = c.x[1];
    }

    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, float> &c, float *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        *(base + offset_) = c.x[0];
        *(base + offset_ + 1) = c.x[1];
    }

    DEVICE_INLINE
    void storeMatrixSyncHalf2M8N8K128(const half2 c, half2 *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4;
        int offset_ = offset + row * ldm + col;
        *(base + offset_) = c;
    }

    // load a matrix [16, 128] rowmajor
    // ldm counts with integer pointers
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<16, 8, 128>> &a, const int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src = base + offset + row * ldm + col;
        a.x[0] = *(uint32_t *)src;
        const int *src2 = base + offset + (row + 8) * ldm + col;
        a.x[1] = *(uint32_t *)src2;
    }

    // load b matrix [128, 8] colmajor = [8, 128] rowmajor
    // ldm counts with integer pointers
    // base data [mma_N, mma_K] rowmajor = [mma_K, mma_N] colmajor
    // So just follow the normal rowmajor thread allocation to read.
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<16, 8, 128>> &b, const int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src = base + offset + row * ldm + col;
        b.x = *(uint32_t *)src;
    }

    // a matrix [16, 128] rowmajor *  b matrix [128, 8] colmajor |  b matrix [8, 128] rowmajor
    DEVICE_INLINE void bmmaSync(fragment_c<ShapeBase<16, 8, 128>, int32_t> &d,
                                const fragment_a_rowmajor<ShapeBase<16, 8, 128>> &a,
                                const fragment_b_colmajor<ShapeBase<16, 8, 128>> &b,
                                const fragment_c<ShapeBase<16, 8, 128>, int32_t> &c)
    {
        ASSEMBLY("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc"
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7,%8,%9,%10};\n"
                : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
                : "r"(a.x[0]), "r"(a.x[1]), "r"(b.x), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
    }

    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<16, 8, 128>, int32_t> &c, int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        *(base + offset_) = c.x[0];
        *(base + offset_ + 1) = c.x[1];
        int offset_2 = offset + (row + 8) * ldm + col;
        *(base + offset_2) = c.x[2];
        *(base + offset_2 + 1) = c.x[3];
    }

    // load a matrix [16, 256] rowmajor
    // ldm counts with integer pointers
    DEVICE_INLINE
    void loadMatrixSync(fragment_a_rowmajor<ShapeBase<16, 8, 256>> &a, const int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src0 = base + offset + row * ldm + col;
        a.x[0] = *(uint32_t *)src0;
        const int *src1 = base + offset + (row + 8) * ldm + col;
        a.x[1] = *(uint32_t *)src1;
        const int *src2 = base + offset + row * ldm + col + 4;
        a.x[2] = *(uint32_t *)src2;
        const int *src3 = base + offset + (row + 8) * ldm + col + 4;
        a.x[3] = *(uint32_t *)src3;
    }

    // load b matrix [128, 8] colmajor = [8, 128] rowmajor
    // ldm counts with integer pointers
    // base data [mma_N, mma_K] rowmajor = [mma_K, mma_N] colmajor
    // So just follow the normal rowmajor thread allocation to read.
    DEVICE_INLINE
    void loadMatrixSync(fragment_b_colmajor<ShapeBase<16, 8, 256>> &b, const int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = lane % 4; // 32 b1 = 1 int
        const int *src0 = base + offset + row * ldm + col;
        b.x[0] = *(uint32_t *)src0;
        const int *src1 = base + offset + row * ldm + col + 4;
        b.x[1] = *(uint32_t *)src1;
    }

    // a matrix [16, 128] rowmajor *  b matrix [128, 8] colmajor |  b matrix [8, 128] rowmajor
    DEVICE_INLINE void bmmaSync(fragment_c<ShapeBase<16, 8, 256>, int32_t> &d,
                                const fragment_a_rowmajor<ShapeBase<16, 8, 256>> &a,
                                const fragment_b_colmajor<ShapeBase<16, 8, 256>> &b,
                                const fragment_c<ShapeBase<16, 8, 256>, int32_t> &c)
    {
        ASSEMBLY("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
                : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]), "r"(b.x[1]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
    }

    DEVICE_INLINE
    void storeMatrixSync(const fragment_c<ShapeBase<16, 8, 256>, int32_t> &c, int *base,
                        const int offset, const int ldm)
    {
        int lane = threadIdx.x & 31;
        int row = lane >> 2;
        int col = (lane % 4) * 2; // // Each thread holds two s32 type data
        int offset_ = offset + row * ldm + col;
        *(base + offset_) = c.x[0];
        *(base + offset_ + 1) = c.x[1];
        int offset_2 = offset + (row + 8) * ldm + col;
        *(base + offset_2) = c.x[2];
        *(base + offset_2 + 1) = c.x[3];
    }
    // #else

    //     assert(false && "bmma is not supported on this architecture( >= 75)\n");

    #endif

} // namespace fq_bmma