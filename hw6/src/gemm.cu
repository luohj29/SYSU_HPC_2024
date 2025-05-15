#include "../include/gemm.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <cassert> // 包含 assert 的头文件
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
// GEMM implementation in global memory
__global__ void gpu_matrix_mult_gm(const float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;

    if (row < m && col < k) // Ensure bounds are within the matrix dimensions
    {
        for (int i = 0; i < n; i++)
        {
            temp += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = temp;
    }
}

// GEMM implementation in shared memory
__global__ void gpu_matrix_mult_sm(const float *a, float *b, float *c, int m, int n, int k)
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;

    int block_num = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // Divide the n dimension by block size
    for (int i = 0; i < block_num; ++i)
    {
        // Copy the ith block into shared memory
        if (row < m && i * BLOCK_SIZE + threadIdx.x < n)
        {
            int a_index = row * n + i * BLOCK_SIZE + threadIdx.x;
            tile_a[threadIdx.y][threadIdx.x] = a[a_index];
        }
        else
            tile_a[threadIdx.y][threadIdx.x] = 0; // Handle edge case for smaller matrix

        if (col < k && i * BLOCK_SIZE + threadIdx.y < n)
        {
            int b_index = (i * BLOCK_SIZE + threadIdx.y) * k + col;
            tile_b[threadIdx.y][threadIdx.x] = b[b_index];
        }
        else
            tile_b[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // Synchronize threads in the block before computation

        // Compute the contribution for c[row][col]
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            temp += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x];
        }
        __syncthreads(); // Ensure all threads finish their computation before moving on
    }
    if (row < m && col < k)
    {
        c[row * k + col] = temp;
    }
}

// Another version of GEMM implementation using shared memory and tiling
__global__ void gpu_matrix_mult_sm_v2(float *a, float *b, float *c, int m, int n, int k)
{
    const int BM = 128;
    const int BN = 8;
    const int BK = 128;
    const int TM = 8;
    const int TN = 8;
    const int Row = blockIdx.y;
    const int Col = blockIdx.x;

    __shared__ float As[BM][BN];
    __shared__ float Bs[BN][BK];

    // Modify pointers for the blocktiles in memory
    a += Row * BM * n;
    b += Col * BK;
    c += Row * BM * n + Col * BK;
}

// GEMM with warp tiling optimization
__global__ void sgemm2DWarpTiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i)
            {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
            {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
    {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}

// CPU implementation of GEMM (General Matrix Multiply)
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void gpu_gemm_cublas(const float *d_a, const float *d_b, float *d_result, int m, int n, int k)
{
    //   m*n x n*k =m *k
    printf("cublas params: %d %d %d\n", m, n, k);
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Configure cuBLAS operations
    const float alpha = 1.0, beta = 0.0;
    cublasOperation_t opA = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t opB = CUBLAS_OP_N; // No transpose for B
    cublasSgemm(handle, opA, opB, k, m, n, &alpha, d_b, k, d_a, n, &beta, d_result, k);
    cublasDestroy(handle);
}

