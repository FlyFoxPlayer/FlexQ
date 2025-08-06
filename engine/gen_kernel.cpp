#include<bits/stdc++.h>
using namespace std;
#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define SCALE_SIZE_X(x) ((x + 3) / 4 * 4)  // Align to 16 bytes
#define SCALE_SIZE_W(x) (x / 2)  // Align to 16 bytes (x is typically BLOCK_N, ensuring 8-byte alignment)

int w_bits = 6;
int a_bits = 6;
int mma_shape[3] = {8, 8, 128};
int GROUP_SIZE = 128;

int main(){
	for(int cta_z = 2 * 128; cta_z <= 512; cta_z += 128)
		for(int WARPS_X = 1; WARPS_X <= 8; WARPS_X = WARPS_X * 2)
			for(int WARPS_Y = 1; WARPS_Y <= 8; WARPS_Y = WARPS_Y * 2){
				if(WARPS_X == 1 && WARPS_Y == 1)continue; // Skip when WARPS_SHAPE equals [1, 1]
				for(int warp_x = mma_shape[0]; warp_x <= 128; warp_x += mma_shape[0])
					for(int warp_y = mma_shape[1]; warp_y <= 128; warp_y += mma_shape[1]){
						// (BLOCK_M * A_BITS) / WARP_M = WARPS_X --> BLOCK_M = (WARPS_X * WARP_M) / A_BITS --> WARP_M = (BLOCK_M * A_BITS) / WARPS_X
						int cta_x = (WARPS_X * warp_x) / a_bits;
						// (BLOCK_N * W_BITS) / WARP_N = WARPS_Y --> BLOCK_N = (WARPS_Y * WARP_N) / W_BITS --> WARP_N = (BLOCK_N * W_BITS) / WARPS_Y
						int cta_y = (WARPS_Y * warp_y) / w_bits;  
						
						// Prune search space
						if((cta_x & cta_x - 1) != 0)continue;
						if(cta_x > 8)continue;
						if(cta_x <= 8 && WARPS_X != 1)continue; 
						if(warp_y != mma_shape[1] * w_bits)continue;
						if(cta_y % 16 != 0)continue;
						
						// Exclude cases where blockDims == 0
						int WARPS_M_NUMS = CEIL(cta_x * a_bits, mma_shape[0]) / (warp_x / mma_shape[0]);
						int WARPS_N_NUMS = CEIL(cta_y * w_bits, mma_shape[1]) / (warp_y / mma_shape[1]);
						int blockDims = 32 * WARPS_M_NUMS * WARPS_N_NUMS;
						if(blockDims == 0)continue;
						
						printf("// cta<%d,%d,%d> warp<%d,%d,128> mma<8,8,128>   WARPS[%dx%d]\n", cta_x, cta_y, cta_z, warp_x, warp_y, WARPS_X, WARPS_Y);
						for(int stage = 2; stage <= 6; stage ++){
							// Exclude configurations exceeding maximum shared memory size
							int BLOCK_M = cta_x;
							int BLOCK_N = cta_y;
							int BLOCK_K = cta_z;
							int MMA_M = mma_shape[0];
							int MMA_N = mma_shape[1];
							int SKEW = w_bits * BLOCK_N % 16 == 0 ? 8 : 0;
	                        int input_buffer_size =
	                            stage * BLOCK_M * BLOCK_K * a_bits / 8 + stage * BLOCK_N * BLOCK_K * w_bits / 8
									+ stage * SCALE_SIZE_X(BLOCK_M) * BLOCK_K / GROUP_SIZE * 4 
									+ stage * SCALE_SIZE_W(BLOCK_N) * BLOCK_K / GROUP_SIZE * 4;
	                        int output_buffer_size = BLOCK_M * (BLOCK_N + 8) * sizeof(int) / 2; 
	                        int shared_mem_size = max(input_buffer_size, output_buffer_size); // bytes
	                        if (shared_mem_size >= 101376) // out of shared memory
                            	continue;
							for(int CTA_TILE_STRIDE = 1; CTA_TILE_STRIDE <= 1; CTA_TILE_STRIDE *= 2){
								// FQ_INSTANTIATE_FUN || FQ_DECL_FUN
								printf("FQ_DECL_FUN(FQBMMA, %d, %d, true, %d, %d, %d, %d, %d, 128, 8, 8, 128, %d, %d);\n", a_bits, w_bits, cta_x, cta_y, cta_z, warp_x, warp_y, stage, CTA_TILE_STRIDE);
							}
						}
						printf("\n");
					 } 
			}
	return 0;
} 
