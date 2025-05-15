#ifndef CONV_H
#define CONV_H
#include "class.cuh"
#include "gemm.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <chrono>
#include <cstring>

// Define block size for the convolution kernel
#define BLOCK_SIZE 32

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

enum CopyType
{
    HostToDevice = 0,
    DeviceToHost = 1
};

/**
 * @brief Perform 2D convolution calculation on device using global memory.
 *
 * @param input  The input matrix in device memory.
 * @param output  The output matrix in device memory.
 * @param output_block  The output block size.
 * @param m  Number of rows of the input matrix.
 * @param n  Number of columns of the input matrix.
 * @param kernelSet  Kernel set containing the kernels for convolution.
 * @param stride  Stride for the convolution.
 * @param padding  Padding value for the convolution.
 */
__global__ void conv2d_cal(float *input, float *output, int output_block, int m, int n, const Kernel kernelSet, int stride, int padding);

/**
 * @brief Pad the input matrix in the given memory, workable in a 3D matrix.
 *
 * @param matrix  The input matrix in linear memory.
 * @param padded_matrix  The output matrix with enough given memory for padding.
 * @param m  Matrix height (number of rows).
 * @param n  Matrix width (number of columns).
 * @param depth  Matrix depth (for 3D matrices).
 * @param padding  The padding parameter to apply.
 * @return float*  The pointer to the padded matrix.
 */
float *padMatrix(const float *matrix, float *padded_matrix, int m, int n, int depth, int padding);

/**
 * @brief Perform 2D convolution with a set of kernels.
 *
 * @param Block  The dimensions of the CUDA block.
 * @param Grid  The dimensions of the CUDA grid.
 * @param input  The input matrix.
 * @param output  The output matrix.
 * @param kernelSet  The kernel set.
 * @param stride  The stride for the convolution.
 * @param padding  The padding for the convolution.
 */
void conv2d(dim3 Block, dim3 Grid, matrixs input, matrixs output, Kernel kernelSet, int stride, int padding);

/**
 * @brief Perform matrix convolution using GEMM (General Matrix Multiply) for convolution.
 *
 * @param Block  The block dimensions for CUDA kernel.
 * @param Grid  The grid dimensions for CUDA kernel.
 * @param h_input  The input matrix.
 * @param h_output  The output matrix.
 * @param kernelSet  The kernel set.
 * @param stride  The stride for the convolution.
 * @param padding  The padding for the convolution.
 */
void conv2d_gemm(dim3 Block, dim3 Grid, matrixs h_input, matrixs h_output, Kernel kernelSet, int stride, int padding);

/**
 * @brief Perform matrix convolution using GEMM (General Matrix Multiply) for convolution.
 *
 * @param Block  The block dimensions for CUDA kernel.
 * @param Grid  The grid dimensions for CUDA kernel.
 * @param h_input  The input matrix.
 * @param h_output  The output matrix.
 * @param kernelSet  The kernel set.
 * @param stride  The stride for the convolution.
 * @param padding  The padding for the convolution.
 */
void conv2d_gemm_v2(dim3 Block, dim3 Grid, matrixs& h_input, matrixs& h_output, Kernel& kernelSet, int stride, int padding);
/**
 * @brief Allocate 3D device memory for matrix storage.
 *
 * @param width  The width (columns) of the matrix.
 * @param height  The height (rows) of the matrix.
 * @param depth  The depth of the matrix.
 * @return cudaPitchedPtr  The allocated 3D matrix on the device.
 */
cudaPitchedPtr allocate3DDeviceMemory(size_t width, size_t height, size_t depth);

/**
 * @brief Copy data between host and device memory for 1D to 3D memory operations.
 *
 * @param host_data  The pointer to host data.
 * @param device_data  The pointer to device data.
 * @param width  The width of the matrix.
 * @param height  The height of the matrix.
 * @param depth  The depth of the matrix.
 * @param cpy_type  The type of copy operation (HostToDevice or DeviceToHost).
 */
void copy1DTo3DDeviceMemory(void *host_data, void *device_data, size_t width, size_t height, size_t depth, CopyType cpy_type);


void conv_cudnn(matrixs h_input, matrixs h_output, Kernel kernelSet, int stride, int padding);
#endif // CONV_H
