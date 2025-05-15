#ifndef GEMM_H
#define GEMM_H

// Define block size for shared memory
#define BLOCK_SIZE 32

/**
 * @brief Implementation of GEMM (General Matrix Multiply) in global memory.
 * 
 * This kernel performs matrix multiplication using global memory.
 * 
 * @param a  Pointer to the first matrix (m x n) in device memory.
 * @param b  Pointer to the second matrix (n x k) in device memory.
 * @param c  Pointer to the result matrix (m x k) in device memory.
 * @param m  Number of rows of the first matrix (a).
 * @param n  Number of columns of the first matrix (a) and rows of the second matrix (b).
 * @param k  Number of columns of the second matrix (b).
 */
__global__ void gpu_matrix_mult_gm(const float *a, float *b, float *c, int m, int n, int k);

/**
 * @brief Implementation of GEMM (General Matrix Multiply) in shared memory.
 * 
 * This kernel performs matrix multiplication using shared memory for optimization.
 * 
 * @param a  Pointer to the first matrix (m x n) in device memory.
 * @param b  Pointer to the second matrix (n x k) in device memory.
 * @param c  Pointer to the result matrix (m x k) in device memory.
 * @param m  Number of rows of the first matrix (a).
 * @param n  Number of columns of the first matrix (a) and rows of the second matrix (b).
 * @param k  Number of columns of the second matrix (b).
 */
__global__ void gpu_matrix_mult_sm(const float *a, float *b, float *c, int m, int n, int k);

/**
 * @brief Another version of GEMM implementation using shared memory and tiling.
 * 
 * @param a  Pointer to the first matrix (m x n) in device memory.
 * @param b  Pointer to the second matrix (n x k) in device memory.
 * @param c  Pointer to the result matrix (m x k) in device memory.
 * @param m  Number of rows of the first matrix (a).
 * @param n  Number of columns of the first matrix (a) and rows of the second matrix (b).
 * @param k  Number of columns of the second matrix (b).
 */
__global__ void gpu_matrix_mult_sm_v2(float *a, float *b, float *c, int m, int n, int k);

/**
 * @brief GEMM kernel with warp tiling optimization.
 * 
 * @param M  Number of rows of the first matrix.
 * @param N  Number of columns of the second matrix.
 * @param K  Number of columns of the first matrix and rows of the second matrix.
 * @param alpha  Scaling factor for the result.
 * @param A  Pointer to the first matrix (M x K) in device memory.
 * @param B  Pointer to the second matrix (K x N) in device memory.
 * @param beta  Scaling factor for the existing result matrix (C).
 * @param C  Pointer to the result matrix (M x N) in device memory.
 */
__global__ void sgemm2DWarpTiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

/**
 * @brief CPU implementation of GEMM.
 * 
 * This function performs matrix multiplication on the CPU.
 * 
 * @param h_a  Pointer to the first matrix (m x n) in host memory.
 * @param h_b  Pointer to the second matrix (n x k) in host memory.
 * @param h_result  Pointer to the result matrix (m x k) in host memory.
 * @param m  Number of rows of the first matrix (a).
 * @param n  Number of columns of the first matrix (a) and rows of the second matrix (b).
 * @param k  Number of columns of the second matrix (b).
 */
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k);

void gpu_gemm_cublas(const float *d_a, const float *d_b, float *d_result, int m, int n, int k);
#endif // GEMM_H
