#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(status)                                   \
    do {                                                      \
        cudaError_t err = status;                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                     \
    } while (0)

#define CHECK_CUBLAS(status)                                  \
    do {                                                      \
        cublasStatus_t err = status;                          \
        if (err != CUBLAS_STATUS_SUCCESS) {                   \
            std::cerr << "cuBLAS error: "                     \
                      << cublasGetStatusString(err)           \
                      << " at line " << __LINE__ << std::endl;\
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#define NUM_PROFILE 1000

// CPU implementation of int8 matrix multiplication
void cpu_int8_gemm(int8_t* A, int8_t* B, int32_t* C, 
                   int m, int n, int k) {
    // Clear output matrix
    for (int i = 0; i < m * n; i++) {
        C[i] = 0;
    }
    
    // Matrix multiplication: C = A * B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t sum = 0;
            for (int l = 0; l < k; l++) {
                // A is row-major (m×k), B is column-major (k×n)
                sum += static_cast<int32_t>(A[i * k + l]) * 
                       static_cast<int32_t>(B[l * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

// Function to verify GPU results using CPU calculation
bool verify_results(int32_t* cpu_c, int32_t* gpu_c, int size) {
    for (int i = 0; i < size; i++) {
        if (cpu_c[i] != gpu_c[i]) {
            std::cerr << "Mismatch at index " << i 
                      << ": CPU=" << cpu_c[i] 
                      << ", GPU=" << gpu_c[i] 
                      << std::endl;
            return false;
        }
    }
    return true;
}

void run_cublas_gemm(int m, int n, int k) {
    // Set matrix dimensions and sizes
    size_t size_A = m * k * sizeof(int8_t);
    size_t size_B = k * n * sizeof(int8_t);
    size_t size_C = m * n * sizeof(int32_t);
    uint64_t seed = 0x2019;

    // Allocate host memory using malloc
    int8_t* h_A = (int8_t*)malloc(size_A);
    int8_t* h_B = (int8_t*)malloc(size_B);
    int32_t* h_C = (int32_t*)malloc(size_C);
    int32_t* h_cpu_C = (int32_t*)malloc(size_C);
    int32_t* h_gpu_C = (int32_t*)malloc(size_C);

    // Initialize host matrices
    srand(seed);
    for (int i = 0; i < m * k; i++) 
        h_A[i] = static_cast<int8_t>(rand() % 256 - 128); // [-128, 127]
    for (int i = 0; i < k * n; i++) 
        h_B[i] = static_cast<int8_t>(rand() % 256 - 128); // [-128, 127]

    // // Compute CPU reference result
    // cpu_int8_gemm(h_A, h_B, h_cpu_C, m, n, k);

    // Allocate device memory
    int8_t *d_A, *d_B;
    int32_t *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Set gemm parameters
    const int32_t alpha = 1;
    const int32_t beta = 0;
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    // cuBLAS dimensions:
    int lda = k; // leading dimension for A 
    int ldb = n; // leading dimension for B
    int ldc = n; // leading dimension for C/D
    
    // Timing measurement
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    for (int i = 0; i < NUM_PROFILE; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, transA, transB,
            n, m, k,
            &alpha,
            d_B, CUDA_R_8I, ldb,
            d_A, CUDA_R_8I, lda,
            &beta,
            d_C, CUDA_R_32I, ldc,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // // Copy GPU result back to host
    // CHECK_CUDA(cudaMemcpy(h_gpu_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // // Verify GPU result against CPU result
    // if (!verify_results(h_cpu_C, h_gpu_C, m * n)){
    //     printf("Results mismatch!");
    //     exit(EXIT_FAILURE);
    // }

    // Calculate performance metrics
    double avg_time = milliseconds / NUM_PROFILE;
    double flops = 2.0 * m * n * k;
    double tflops = (flops * 1e-12) / (avg_time / 1000.0);

    printf("cuBLAS-W8A8-GEMM. m: %6d, n: %6d, k: %6d,\t Time: %.4f ms, TFLOPS: %4.4f\n",
           m, n, k, avg_time, tflops);

    // Clean up resources
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_cpu_C);
    free(h_gpu_C);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        printf("Example: %s 1 4096 4096\n", argv[0]);
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Verify GPU compute capability
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    
    if (props.major < 7) {
        std::cerr << "cuBLAS int8 GEMM requires compute capability 70+ (Volta+)\n";
        return 0;
    }

    run_cublas_gemm(m, n, k);
    return 0;
}