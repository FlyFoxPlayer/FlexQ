#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include "common/timer.h"
#include "src/pack/bit_packing.h"
#include "common/base.h"

// Define row/column major flag
#define ROW_FIRST 0
#define COL_FIRST 1

#define bit_pow(x) (1 << x)

using namespace std;

// Generate random data
template<typename T>
void randomInitMatrix(T* &A, int rows, int cols, int bits, int major = ROW_FIRST) {
    std::srand(time(0));

    // Row-major layout
    if(major == ROW_FIRST){
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++){
                A[i * cols + j] = static_cast<T>((T)std::rand() % bit_pow(bits));
            }
        }
    }

    // Column-major layout  
    if(major == COL_FIRST){
        for(int i = 0; i < cols; i++) {
            for(int j=0; j < rows; j++){
                A[j * cols + i] = static_cast<T>((T)std::rand() % bit_pow(bits));
            }
        }
    }
}

/*
x: [m, k] -> [x_bit * m * k / 32]
w: [n, k] -> [w_bit * n * k / 32]
*/
int main(int argc, char **argv){
    if (argc < 4) {
        printf("Usage: ./test_packing_kernel M K X_BITS\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int x_bits = atoi(argv[3]);

    if (k < 128 || k % 128 != 0) {
        printf("Unsupported computational layout! k must >= 128 and k %% 128 == 0!\n");
        return -1;
    }

    int repeat = 1000;
    int warmup = 10;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate host memory
    int *h_x = (int *)malloc(m * k * sizeof(int));
    int *h_x_pack_abq = (int *)malloc(x_bits * m * (k / 32) * sizeof(int));
    int *h_x_pack_flexq_update_final = (int *)malloc((k / 128) * m * x_bits * 4 * sizeof(int));

    // Allocate device memory
    int *d_x;
    int *d_x_pack_abq;
    int *d_x_pack_flexq_update_final;
    cudaMalloc(&d_x, m * k * sizeof(int));
    cudaMalloc(&d_x_pack_abq, x_bits * m * (k / 32) * sizeof(int));
    cudaMalloc(&d_x_pack_flexq_update_final, (k / 128) * m * x_bits * 4 * sizeof(int));

    randomInitMatrix(h_x, m, k, x_bits);
    cudaMemcpy(d_x, h_x, sizeof(int) * m * k, cudaMemcpyHostToDevice);

    if (x_bits <= 32) {
        abq_bit_packing(d_x, d_x_pack_abq, m, k, x_bits, stream);
        flexq_bit_packing(d_x, d_x_pack_flexq_update_final, m, k, x_bits, stream);
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

    // ABQ packing kernel
    CudaTimer abq_exec_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            abq_exec_timer.start();
        abq_bit_packing(d_x, d_x_pack_abq, m, k, x_bits, stream);
    }
    abq_exec_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "abq packing kernel failed." << std::endl;
        return -1;
    }
    float abq_exec_dur = abq_exec_timer.elapsed_msecs() / repeat;

    // FlexQ update final packing kernel
    CudaTimer flexq_update_final_exec_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            flexq_update_final_exec_timer.start();
        flexq_bit_packing(d_x, d_x_pack_flexq_update_final, m, k, x_bits, stream);
    }
    flexq_update_final_exec_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "flexq update final packing kernel failed." << std::endl;
        return -1;
    }
    float flexq_update_final_exec_dur = flexq_update_final_exec_timer.elapsed_msecs() / repeat;


    cudaMemcpy(h_x_pack_abq, d_x_pack_abq, x_bits * m * (k / 32) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_pack_flexq_update_final, d_x_pack_flexq_update_final, (k / 128) * m * x_bits * 4 * sizeof(int), cudaMemcpyDeviceToHost);


    // Validate FlexQ packing kernel correctness
    bool flag = true;
    const int chunk_m = min(m, 8);
    for(int bit = 0; bit < x_bits; bit++){
        for(int iter_m = 0; iter_m < m; iter_m++){
            const int chunk_m_id = iter_m / chunk_m;
            const int chunk_m_row = iter_m % chunk_m;
            for(int iter_k = 0; iter_k < k / 32; iter_k++){
                // [x_bits, m, k / 32]  --> [m, k / 128, x_bits * 4]
                if(h_x_pack_abq[bit * (m * k / 32) + iter_m * (k / 32) + iter_k] != 
                        h_x_pack_flexq_update_final[(iter_k / 4) * (m * x_bits * 4) + chunk_m_id * (x_bits * chunk_m * 4) + bit * (chunk_m * 4) + chunk_m_row * 4 + iter_k % 4]){
                    flag = false;
                }
            }
        }
    }
    if(!flag)cout << "FlexQ update Packing final kernel ERROR! Inconsistent results!" << endl;
    else cout << "FlexQ update Packing final kernel SUCCESS! consistent results!" << endl;

    
    printf("\nKernel performance:\n");
    printf("ABQ packing %f (us) exec\n", abq_exec_dur * 1e3);  
    printf("FlexQ update final packing %f (us) exec\n", flexq_update_final_exec_dur * 1e3);                                     

    free(h_x);
    free(h_x_pack_abq);
    free(h_x_pack_flexq_update_final);
    cudaFree(d_x);
    cudaFree(d_x_pack_abq);
    cudaFree(d_x_pack_flexq_update_final);

    cudaStreamDestroy(stream);

    return 0;
}