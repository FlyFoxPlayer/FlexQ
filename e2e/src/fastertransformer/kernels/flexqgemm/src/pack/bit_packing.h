#pragma once
#include "src/fastertransformer/kernels/flexqgemm/common/device.h"
#include "src/fastertransformer/kernels/flexqgemm/common/base.h"
#include <cuda_fp16.h>

#define BITS_INT 32     // Number of bits in an int 
#define BITS_INT4 128     // Number of bits in an int4
#define INT4_NUIT 4     // Number of int4 elements 

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

//get lane id
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid)); 
//get warp id
#define GET_WARPID unsigned warpid; asm("mov.u32 %0, %%warpid;":"=r"(warpid)); 

void flexq_bit_packing(const int* in_data, int* packed_data, const int M, const int K, const int BIT, cudaStream_t stream);

void flexq_bit_packing(const half* in_data, int* packed_data, half* T_out_scale, const int M, const int K, const int BIT, cudaStream_t stream);