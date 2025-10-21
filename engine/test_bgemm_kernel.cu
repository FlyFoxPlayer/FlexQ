#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include "src/pack/bit_packing.h"
#include "common/base.h"
#include "test/test_kernel.h"

// Define row/column major flag
#define ROW_FIRST 0
#define COL_FIRST 1

#define bit_pow(x) (1 << x)

using namespace std;

template<typename T>
void randomInitMatrix(T* &A, int rows, int cols, int bits, int major = ROW_FIRST) {
    // row first
    if(major == ROW_FIRST){
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++){
                A[i * cols + j] = (T)(std::rand() % bit_pow(bits));
            }
        }
    }

    // col first
    if(major == COL_FIRST){
        for(int i = 0; i < cols; i++) {
            for(int j=0; j < rows; j++){
                A[j * cols + i] = (T)(std::rand() % bit_pow(bits));
            }
        }
    }
}

// ROW_FIRST 
void randomInitHalfASCALE(half* &A, int rows, int cols){
    int row_offset = SCALE_PACKING_A(SCALE_SIZE_X(cols));

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < row_offset; j+=2){
            if(j < SCALE_PACKING_A(cols)){
                half temp = __float2half(0.1f * rand() / RAND_MAX);
                A[i * row_offset + j] = A[i * row_offset + j + 1] = temp;
            }else{
                A[i * row_offset + j] = A[i * row_offset + j + 1] = 0.0f;
            }
        }
    }
}

// ROW_FIRST 
void randomInitHalfBSCALE(half* &B, int rows, int cols){
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++){
            B[i * cols + j] = __float2half(0.1f * rand() / RAND_MAX);  
        }
    }
}

// Verify the correctness of the computation
bool check_2D_half(half* A, half* B, int m, int n){
    try{
        int max_error_nums = 5; // For debugging only - displays the location of the error.
        bool correct = true;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                float dev = fabs(__half2float(A[i * n + j]) - __half2float(B[i * n + j]));
                // FP16 (half-precision) computations inherently introduce numerical deviations compared to FP32 (float)
                // Error tolerance within 0.01% of half precision range
                if(dev > 0.0001 * HALF_MAX_RANGE){
                    correct = false;
                    if(max_error_nums > 0){
                        max_error_nums --;
                        // printf("result error:GPU[%d, %d]:%f CPU[%d, %d]:%f\n", i, j, __half2float(A[i * n + j]), i, j, __half2float(B[i * n + j]));
                    }
                }
            }
        }
        if(correct){
            return 1;
        }else return 0;
        
    }catch(std::exception& e){
        std::cout << e.what() << std::endl;
        return 0;
    }
}

// Binary exponentiation
int int_pow(int base, int exp)
{
    int result = 1;
    while (exp) {
        if (exp % 2)
            result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

/*
x: [M, K / 32, x_bits]
w: [N, K / 32, w_bits]
x_scale: [K / group_size, SCALE_SIZE_A(M)] --> layout: (half2)[(half)element 0, (half)element 0]
w_scale: [K / group_size, SCALE_SIZE_B(N)] --> layout: (half2)[(half)element 0, (half)element 1]
*/
void compute_ref(int *w, half* w_scale, int *x, half* x_scale, half *ref_c, int M, int N, int K, int W_BIT, int X_BIT, bool SIGNED, int group_size)
{
    const int MMA_M = 8;
    const int MMA_N = 8;
    int chunk_m = min(M, MMA_M);
    int chunk_n = min(N, MMA_N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float tmp = 0;
            for (int xb = 0; xb < X_BIT; xb++) {
                int X_Multiplier =
                    SIGNED && (xb == X_BIT - 1) ? -1 * int_pow(2, xb) : int_pow(2, xb);
                for (int wb = 0; wb < W_BIT; wb++) {
                    int W_Multiplier =
                        SIGNED && (wb == W_BIT - 1) ? -1 * int_pow(2, wb) : int_pow(2, wb);
                    for (int k_tile = 0; k_tile < K / 32; k_tile++) {
                        int w_int = w[(k_tile / 4) * (N * W_BIT * 4) + (n / chunk_n) * (W_BIT * chunk_n * 4) + wb * (chunk_n * 4) + (n % chunk_n) * 4 + (k_tile % 4)];
                        int x_int = x[(k_tile / 4) * (M * X_BIT * 4) + (m / chunk_m) * (X_BIT * chunk_m * 4) + xb * (chunk_m * 4) + (m % chunk_m) * 4 + (k_tile % 4)];
                        float w_scale_float = __half2float(w_scale[k_tile / (group_size / 32) * N + n]);
                        float x_scale_float = __half2float(x_scale[k_tile / (group_size / 32) * SCALE_PACKING_A(SCALE_SIZE_X(M)) + 2 * m]);

                        for (int k = 0; k < 32; k++) {
                            int mask = 1;
                            int x_val = ((mask << k) & x_int) >> k;
                            int w_val = ((mask << k) & w_int) >> k;
                            tmp += 1.0 * (X_Multiplier * W_Multiplier * x_val * w_val) * w_scale_float * x_scale_float;
                        }
                    }
                }
            }
            ref_c[m * N + n] = __float2half(tmp);
        }
    }
}

int main(int argc, char **argv){
    if (argc < 6) {
        printf("Usage: ./test_bgemm_kernel M N K X_BITS W_BITS\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int x_bits = atoi(argv[4]);
    int w_bits = atoi(argv[5]);

    bool debug_flag = false;
    if(argc >= 7 && atoi(argv[6]) == 1){
        debug_flag = true;
    }

    const bool quant_sign = 1;
    const int group_size = 128;

    if (group_size < 32 || group_size % 32 != 0) {
        printf("Unsupported group_size! group_size must >= 32 and group_size %% 32 == 0!\n");
        return -1;
    }

    if (k < 128 || k % 128 != 0) {
        printf("Unsupported computational layout! k must >= 128 and k %% 128 == 0!\n");
        return -1;
    }

    std::srand(time(0));

    int repeat = 1000;
    int warmup = 10;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate host memory space
    int *h_x = (int *)malloc(m * k * sizeof(int));  // host activation input
    int *h_w = (int *)malloc(n * k * sizeof(int));  // host weight input
    half *h_x_scale = (half *)malloc(k / group_size * SCALE_PACKING_A(SCALE_SIZE_X(m)) * sizeof(half));  // host activation input scale [K / GROUP_SIZE, M]
    half *h_w_scale = (half *)malloc(k / group_size * n * sizeof(half));  // host weight input scale [K / GROUP_SIZE, N]
    int *h_x_pack = (int *)malloc(x_bits * m * (k / 32) * sizeof(int));  // host activation input(packing)
    int *h_w_pack = (int *)malloc(w_bits * n * (k / 32) * sizeof(int));  // host weight input(packing)
    half *h_out = (half *)malloc(m * n * sizeof(half));  // host output(GPU)
    half *h_ref_out = (half *)malloc(m * n * sizeof(half));  // host output(CPU)

    // Allocate device memory space
    int *d_x;
    half *d_x_scale;
    int *d_x_pack;
    int *d_w;
    half *d_w_scale;
    int *d_w_pack;
    half *d_out;
    cudaMalloc(&d_x, m * k * sizeof(int));
    cudaMalloc(&d_w, n * k * sizeof(int));
    cudaMalloc(&d_x_scale, k / group_size * SCALE_PACKING_A(SCALE_SIZE_X(m)) * sizeof(half));
    cudaMalloc(&d_w_scale, k / group_size * n * sizeof(half));
    cudaMalloc(&d_x_pack, x_bits * m * (k / 32) * sizeof(int));
    cudaMalloc(&d_w_pack, w_bits * n * (k / 32) * sizeof(int));
    cudaMalloc(&d_out, m * n * sizeof(half));

    // INIT HOST TENSOR
    randomInitMatrix(h_x, m, k, x_bits);
    randomInitMatrix(h_w, n, k, w_bits);
    randomInitHalfASCALE(h_x_scale, k / group_size, m);
    randomInitHalfBSCALE(h_w_scale, k / group_size, n);
    cudaMemcpy(d_x, h_x, sizeof(int) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_scale, h_x_scale, sizeof(half) * k / group_size * SCALE_PACKING_A(SCALE_SIZE_X(m)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_scale, h_w_scale, sizeof(half) * k / group_size * n, cudaMemcpyHostToDevice);

    if (w_bits <= 32) {
        flexq_bit_packing(d_w, d_w_pack, n, k, w_bits, stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Line %d: 'weight bit_pack' failed: %s\n", __LINE__, cudaGetErrorString(err));
            return -1;
        }
    } else {
        printf("unsupport w_bits %d: for bit_pack func \n", w_bits);
        return -1;
    }

    if (x_bits <= 32) {
        flexq_bit_packing(d_x, d_x_pack, m, k, x_bits, stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Line %d: 'activation bit_pack' failed: %s\n", __LINE__,
                   cudaGetErrorString(err));
            return -1;
        }
    } else {
        printf("unsupport x_bits %d: for bit_pack func \n", x_bits);
        return -1;
    }

    cudaMemcpy(h_x_pack, d_x_pack, x_bits * m * (k / 32) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w_pack, d_w_pack, w_bits * n * (k / 32) * sizeof(int), cudaMemcpyDeviceToHost);

    compute_ref(h_w_pack, h_w_scale, h_x_pack, h_x_scale, h_ref_out, m, n, k, w_bits, x_bits, quant_sign, group_size);

    switch (w_bits) {
        case 6:
            switch (x_bits) {
                case 6:
                    printf("test_w6a6_kernel\n");
                    test_w6a6_kernel(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, d_x_scale, d_w_scale, m, n, k, d_out, h_out, h_ref_out, warmup, repeat, quant_sign, group_size, stream);
                    break;
                case 8:
                    printf("test_w6a8_kernel\n");
                    test_w6a8_kernel(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, d_x_scale, d_w_scale, m, n, k, d_out, h_out, h_ref_out, warmup, repeat, quant_sign, group_size, stream);
                    break;
                default:
                    printf("unsupport w%da%d!\n", w_bits, x_bits);
            }
            break;
        default:
            printf("unsupport w%da%d!\n", w_bits, x_bits);
    }

    cudaMemcpy(h_out, d_out, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    bool flag = check_2D_half(h_out, h_ref_out, m, n);

    if(!flag)cout << "ERROR! Inconsistent results!" << endl;
    else cout << "SUCCESS! consistent results!" << endl;

    free(h_x);
    free(h_w);
    free(h_x_pack);
    free(h_w_pack);
    free(h_x_scale);
    free(h_w_scale);
    free(h_out);
    free(h_ref_out);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_x_pack);
    cudaFree(d_w_pack);
    cudaFree(d_x_scale);
    cudaFree(d_w_scale);
    cudaFree(d_out);

    cudaStreamDestroy(stream);

    return 0;
}