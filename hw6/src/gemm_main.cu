#include "../include/gemm.cuh"
#include "../include/utilis.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k>\n";
        return 1;
    }
    // get_device_info();
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    int block_szie = atoi(argv[4]);

    // Initialize cuBLAS


    // Scalar coefficients
    const float alpha = 1.0f, beta = 0.0f;

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    cudaHostAlloc((void **)&h_A, sizeof(float) * m * k, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, sizeof(float) * k * n, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, sizeof(float) * m * n, cudaHostAllocDefault);

    // Initialize matrices with random values
    randomize_matrix(h_A, m, k);
    randomize_matrix(h_B, k, n);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * m * k);
    cudaMalloc((void **)&d_B, sizeof(float) * k * n);
    cudaMalloc((void **)&d_C, sizeof(float) * m * n);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // Configure cuBLAS operations
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t opA = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t opB = CUBLAS_OP_N; // No transpose for B
    std::chrono::duration<float, std::milli> duration;

    // ----------------------------Measure GPU computation time (custom kernel)------------------------------
    printf("\nTestting the diy_cuda_global_mem implementation of matrix multiply\n");
    dim3 dimBlock(block_szie, block_szie);
    dim3 dimGrid((n + block_szie - 1) / block_szie, (m + block_szie - 1) / block_szie);

    auto start = std::chrono::high_resolution_clock::now();
    gpu_matrix_mult_gm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);  
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    // printf_matrix_corner(h_C, m, n);
    duration = end - start;

    double custom_time = duration.count();
    double custom_gflops = (2.0 * m * n * k) / (custom_time * 1e6);
    printf("Custom Kernel Time: %f ms\tPerformance: %f GFLOPS\n", custom_time, custom_gflops);

    // ----------------------------Measure GPU computation time (custom kernel)------------------------------
    printf("\nTestting the diy_cuda_shared_mem implementation of matrix multiply\n");

    start = std::chrono::high_resolution_clock::now();
    gpu_matrix_mult_sm<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    // printf_matrix_corner(h_C, m, n);
    duration = end - start;

    custom_time = duration.count();
    custom_gflops = (2.0 * m * n * k) / (custom_time * 1e6);
    printf("Custom Kernel Time: %f ms\tPerformance: %f GFLOPS\n", custom_time, custom_gflops);

    // ----------------------------Measure GPU computation time (cuBLAS)---------------------------
    printf("\nTestting the cublas implementation of matrix multiply\n");
    start = std::chrono::high_resolution_clock::now();

    cublasSgemm(handle, opA, opB, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n); // 我日，cublas 居然是列优先的
    // gpu_gemm_cublas(h_A, h_B, h_C, m, k, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    // printf_matrix_corner(h_C, m, n);

    double cublas_time = duration.count();
    double cublas_gflops = (2.0 * m * n * k) / (cublas_time * 1e6);
    printf("cuBLAS Time: %f ms\tPerformance: %f GFLOPS\n", cublas_time, cublas_gflops);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cublasDestroy(handle);
    // --------------------------------Measure CPU computation time----------------------------------------------
    // printf("\nTestting the cpu implementation of matrix multiply\n");
    // start = std::chrono::high_resolution_clock::now();
    // cpu_matrix_mult(h_A, h_B, h_C, m, k, n);
    // printf_matrix_corner(h_C, m, n);
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start;

    // double cpu_time = duration.count();
    // double cpu_gflops = (2.0 * m * n * k) / (cpu_time * 1e6);
    // printf("CPU Time: %f ms\tPerformance: %f GFLOPS\n", cpu_time, cpu_gflops);

    // Cleanup


    return 0;
}
