
#include "../include/conv.cuh"

/*
    description:this function works in the device, each thread deal with the padded matrix to sum up the conv local result
                and map the result to the output matrix
    params: input:padded matrix in dmem, it is allocaated in linear memoey;
            output:output allocated matrix in demem, same in linear memory;
            m,n:    origin matrix before padded
            kernelset: including the kernel info
            strde, padding: conv params
*/
__global__ void conv2d_cal(float *input, float *output, int output_block, int m, int n, const Kernel kernelSet, int stride, int padding)
{
    // return;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int paddedm = m + 2 * padding;
    int paddedn = n + 2 * padding;

    int conved_n = (n - kernelSet.cols + 2 * padding) / stride + 1;

    float temp = 0;

    if (row % stride == 0 && col % stride == 0 && row < paddedm - kernelSet.rows + 1 && col < paddedn - kernelSet.cols + 1)
    {
        for (int k = 0; k < kernelSet.numKernels; k++)
        {
            for (int i = 0; i < kernelSet.rows; i++)
            {
                for (int j = 0; j < kernelSet.cols; j++)
                {
                    int input_row = row + i;
                    int input_col = col + j;
                    int kernel_idx = k * kernelSet.rows * kernelSet.cols + i * kernelSet.cols + j;
                    assert(kernel_idx < paddedm * paddedn);
                    temp += input[input_row * paddedn + input_col] * kernelSet.deviceKernels[kernel_idx];
                }
            }
        }
        output[(row) / stride * conved_n + (col) / stride] = temp;
    }
}

/**
 * @brief padding the matrix input in the given memory, workable in 3d marix
 *
 * @param matrix  the input matrix in linear memory
 * @param padded_matrix  the ouput matrix with enough given memory
 * @param m  matrix heigth(row)
 * @param n  matrix width(col)
 * @param depth  matrix depth
 * @param padding  the padding para
 * @return float*  actually the given matrix ptr
 */
float *padMatrix(const float *matrix, float *padded_matrix, int m, int n, int depth, int padding)
{
    int rows = m;
    int cols = n;
    int paddedCols = n + 2 * padding;
    int paddedRows = m + 2 * padding;
    int paddedBlcok = paddedRows * paddedCols;
    int originblock = rows * cols;
    for (int k = 0; k < depth; k++)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                padded_matrix[k * paddedBlcok + (i + padding) * paddedCols + j + padding] = matrix[k * originblock + i * n + j];
            }
        }
    }
    return padded_matrix;
}

/*
    description:
    - input and output are both in host mem
    - make the device mem, use the con2d_cal to finish the computation part
*/
void conv2d(dim3 Block, dim3 Grid, matrixs input, matrixs output, Kernel kernelSet, int stride, int padding)
{
    float *d_input;
    float *h_input = input.h_input;
    float *h_output = output.h_input;
    int m = input.rows;
    int n = input.cols;
    int paddedRows = m + 2 * padding;
    int paddedCols = n + 2 * padding;
    float *temp;
    int x = input.matrix_nums;

    assert(input.matrix_nums == kernelSet.numKernels); // 通道数量必须相同

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 计时：内存分配和填充
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMallocHost((void **)&temp, sizeof(float) * paddedRows * paddedCols * input.matrix_nums));
    CUDA_CHECK(cudaMalloc((void **)&d_input, sizeof(float) * paddedRows * paddedCols * x));
    memset(temp, 0, sizeof(float) * paddedRows * paddedCols * input.matrix_nums);
    padMatrix(h_input, temp, m, n, input.matrix_nums, padding);
    CUDA_CHECK(cudaMemcpy(d_input, temp, sizeof(float) * paddedRows * paddedCols * input.matrix_nums, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(temp));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float mem_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&mem_time, start, stop));
    std::cout << "Memory allocation and padding time: " << mem_time << " ms" << std::endl;

    // 计时：卷积核数据拷贝
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMalloc((void **)&temp, sizeof(float) * kernelSet.rows * kernelSet.cols * input.matrix_nums));
    CUDA_CHECK(cudaMemcpy(temp, kernelSet.deviceKernels, sizeof(float) * kernelSet.rows * kernelSet.cols * input.matrix_nums, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernel_copy_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_copy_time, start, stop));
    std::cout << "Kernel data copy time: " << kernel_copy_time << " ms" << std::endl;

    // 计时：卷积计算
    float *d_output;
    int conved_m = (m - kernelSet.rows + 2 * padding) / stride + 1;
    int conved_n = (n - kernelSet.cols + 2 * padding) / stride + 1;
    int output_block = conved_m * conved_n;
    CUDA_CHECK(cudaMalloc((void **)&d_output, sizeof(float) * output_block));

    CUDA_CHECK(cudaEventRecord(start));
    conv2d_cal<<<Grid, Block>>>(d_input, d_output, output_block, m, n, kernelSet, stride, padding);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_POST_KERNEL_CHECK;
    float conv_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&conv_time, start, stop));
    std::cout << "Convolution computation time: " << conv_time << " ms" << std::endl;

    // 计时：结果拷贝回主机
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(float) * output_block, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float memcpy_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_time, start, stop));
    std::cout << "Result copy back to host time: " << memcpy_time << " ms" << std::endl;

    // 释放资源
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return;
}
/*
    description: GEMM version convolution
    - input and output are both in host mem
    - make the device mem, use the con2d_cal to finish the computation part
*/
/*
example: padding =1 ,stride =1(default), here we igonre the other 2 matrix(we have 3 in total!!!)
1 2 3                       1 1                     0 0 0 0 0
4 5 6  input matrix,        1 1  filter/window      0 1 2 3 0 padded matrix
7 8 9                                               0 4 5 6 0
                                                    0 7 8 9 0
                                                    0 0 0 0 0

converted matrix:  convable_m = convable_n = (3+2-2)/1 +1 = 4;
                    converted_n = 4*4 = 16, converted_m = 3 * 2 * 2 = 12;

                converted_n
converted_m     0 0 0 0 0 ......
                0 0 0 0 1
                0 1 2 3 0
                1 2 3 0 4

loop            1 2 3 4 5(i=0, j=1) ......
*/


#include "cuda_runtime_api.h"

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype *data_im,
                                  const int height, const int width, const int ksize, const int pad,
                                  const int stride, const int height_col, const int width_col,
                                  Dtype *data_col)
{
    CUDA_KERNEL_LOOP(index, n) // 宏函数,为每一个index启动线程
    {
        int w_out = index % width_col; // 计算线程对应的输出元素的列索引
        int h_index = index / width_col;
        int h_out = h_index % height_col;      // 计算线程对应的输出元素的行索引
        int channel_in = h_index / height_col; // 计算线程对应的输出元素的通道位置
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad; // 反向计算im2col映射前的元素位置
        Dtype *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out; // 获取映射后的指针位置
        const Dtype *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in; // 映射前的位置
        for (int i = 0; i < ksize; ++i)
        {
            for (int j = 0; j < ksize; ++j)
            {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template <typename Dtype>
void im2col_gpu(const Dtype *data_im, const int channels,
                const int height, const int width, const int ksize, const int pad,
                const int stride, Dtype *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                               CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, height, width, ksize, pad, stride, height_col,
        width_col, data_col);
    cudaDeviceSynchronize();
    // CUDA_POST_KERNEL_CHECK;
}

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void conv2d_gemm_v2(dim3 Block, dim3 Grid, matrixs &h_input, matrixs &h_output, Kernel &kernelSet, int stride, int padding)
{
    int convable_m = (h_input.rows + 2 * padding - kernelSet.rows) / stride + 1;
    int convable_n = (h_input.cols + 2 * padding - kernelSet.cols) / stride + 1;

    int converted_n = convable_m * convable_n;
    int converted_m = kernelSet.rows * kernelSet.cols;

    // 创建 CUDA 事件
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // 创建 CPU 计时器
    auto start_cpu = std::chrono::high_resolution_clock::now();

    float *d_input, *d_matrix, *d_result;

    // 计时：GPU 内存分配和数据拷贝
    cudaEventRecord(start_gpu);
    cudaMalloc((void **)&d_matrix, converted_n * converted_m * h_input.matrix_nums * sizeof(float));
    cudaMalloc((void **)&d_input, sizeof(float) * h_input.rows * h_input.cols * 3);
    cudaMemcpy(d_input, h_input.h_input, sizeof(float) * h_input.rows * h_input.cols * 3, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_mem_time = 0;
    cudaEventElapsedTime(&gpu_mem_time, start_gpu, stop_gpu);
    printf("GPU memory allocation and copy time: %f ms\n", gpu_mem_time);

    // 计时：im2col_gpu（GPU 时间）
    cudaEventRecord(start_gpu);
    im2col_gpu<float>(d_input, 3, h_input.rows, h_input.cols, 3, padding, stride, d_matrix);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_im2col_time = 0;
    cudaEventElapsedTime(&gpu_im2col_time, start_gpu, stop_gpu);
    printf("GPU im2col_gpu time: %f ms\n", gpu_im2col_time);

    // 计时：cublasSgemm（GPU 时间）
    
    int filter_size = kernelSet.cols * kernelSet.rows;
    cudaMalloc((void **)&d_result, sizeof(float) * converted_n);
    cudaEventRecord(start_gpu);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t opA = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t opB = CUBLAS_OP_N; // No transpose for B
    const float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_malloc_time = 0;
    cudaEventElapsedTime(&gpu_malloc_time, start_gpu, stop_gpu);
    printf("GPU cublas handle malloc time: %f ms\n", gpu_malloc_time);

    cudaEventRecord(start_gpu);
    cublasSgemm(handle, opA, opB, converted_n, 1, filter_size * kernelSet.numKernels, &alpha, d_matrix, converted_n, kernelSet.deviceKernels, filter_size * kernelSet.numKernels, &beta, d_result, converted_n);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_gemm_time = 0;
    cudaEventElapsedTime(&gpu_gemm_time, start_gpu, stop_gpu);
    printf("GPU cublasSgemm time: %f ms\n", gpu_gemm_time);

    // 计时：结果拷贝回主机（GPU 时间）
    cudaEventRecord(start_gpu);
    cudaMemcpy(h_output.h_input, d_result, sizeof(float) * converted_n, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_memcpy_time = 0;
    cudaEventElapsedTime(&gpu_memcpy_time, start_gpu, stop_gpu);
    printf("GPU result copy back to host time: %f ms\n", gpu_memcpy_time);

    // 计算 CPU 总时间
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    printf("Total CPU time: %lld ms\n", cpu_duration.count());

    // 释放资源
    cudaFree(d_matrix);
    cudaFree(d_result);
    cublasDestroy(handle);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return;
}
void conv2d_gemm(dim3 Block, dim3 Grid, matrixs h_input, matrixs h_output, Kernel kernelSet, int stride, int padding)
{
    int padded_m = h_input.rows + 2 * padding;
    int padded_n = h_input.cols + 2 * padding;
    int convable_m = (h_input.rows + 2 * padding - kernelSet.rows) / stride + 1;
    int convable_n = (h_input.cols + 2 * padding - kernelSet.cols) / stride + 1;

    int converted_n = convable_m * convable_n;
    int converted_m = kernelSet.rows * kernelSet.cols;

    int block = converted_m * converted_n;
    int padded_block = padded_m * padded_n;

    float *temp;
    cudaMallocHost((void **)&temp, sizeof(float) * padded_block * h_input.matrix_nums); // create a temp to get the padded matrix in host mem

    padMatrix(h_input.h_input, temp, h_input.rows, h_input.cols, h_input.matrix_nums, padding);

    float *h_temp;
    cudaMallocHost((void **)&h_temp, sizeof(float) * h_input.matrix_nums * block); // for converted matrix
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i + kernelSet.rows - 1 < padded_m; i += stride)                // the next 2 iterations are for sliding the conv window
    {
        for (int j = 0; j + kernelSet.cols - 1 < padded_n; j += stride)
        {
            for (int a = 0; a < kernelSet.rows; a++) // the naxt 2 iterations are for inside the conv window
            {
                for (int b = 0; b < kernelSet.cols; b++)
                {
                    for (int k = 0; k < h_input.matrix_nums; k++) // the k iteration for the input matrixs(could be more than 1)
                    {
                        int li = k * block + (a * kernelSet.cols + b) * converted_n + (i / stride) * convable_n + (j / stride);
                        int ri = k * padded_block + (i + a) * padded_m + j + b;
                        h_temp[li] = temp[ri];
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end - start);
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    printf("convertting used time: %fms\n", time);
    cudaFreeHost(temp);
    float *d_matrix, *d_result;
    cudaMalloc((void **)&d_matrix, converted_n * converted_m * h_input.matrix_nums * sizeof(float));
    cudaMemcpy(d_matrix, h_temp, converted_n * converted_m * h_input.matrix_nums * sizeof(float), cudaMemcpyHostToDevice);

    cudaFreeHost(h_temp);

    int filter_size = kernelSet.cols * kernelSet.rows;

    cudaMalloc((void **)&d_result, sizeof(float) * converted_n);
    start = std::chrono::high_resolution_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t opA = CUBLAS_OP_N; // No transpose for A
    cublasOperation_t opB = CUBLAS_OP_N; // No transpose for B
    const float alpha = 1.0f, beta = 0.0f;
    // gpu_gemm_cublas(kernelSet.deviceKernels, d_matrix, d_result, 1, filter_size * kernelSet.numKernels, converted_n);
    cublasSgemm(handle, opA, opB, converted_n, 1, filter_size * kernelSet.numKernels, &alpha, d_matrix, converted_n, kernelSet.deviceKernels, filter_size * kernelSet.numKernels, &beta, d_result, converted_n);
    cudaDeviceSynchronize(); // 确保所有 CUDA 操作完成

    CUDA_POST_KERNEL_CHECK;
    cudaMemcpy(h_output.h_input, d_result, sizeof(float) * converted_n, cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    duration = (end - start);
    time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    printf("conv gemm used time: %f ms\n", time);
    cudaFree(d_matrix);
    cudaFree(d_result);
    cublasDestroy(handle);
    return;
}



// #include "cudnn.h"

#define CHECK_CUDNN_ERR(expression)                                \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

/*
    description: cudnn version convolution
    input: matrixs class h_input is matrixs input; h_output is matrixs output, both in host mem
        kernelSet is in device mem. stride and padding are easy as you see.
*/
/*
    h_input是包含3个RGB矩阵的结构体，h_output是包含3个输出矩阵的结构体，kernelSet是包含3个卷积核（分别对应三个RGB矩阵）的结构体
*/

#include <cuda_runtime.h>
#include <cudnn.h>
void conv_cudnn(matrixs h_input, matrixs h_output, Kernel kernelSet, int stride, int padding)
{
    int N = 1; // Batch size
    int C = h_input.matrix_nums;
    int H = h_input.rows;
    int W = h_input.cols;

    int R = kernelSet.rows;
    int S = kernelSet.cols;

    int conved_H = h_output.rows;
    int conved_W = h_output.cols;

    // Check if the kernel count matches the input channels
    if (kernelSet.numKernels != h_input.matrix_nums)
    {
        printf("Err: not paired kernels and input\n");
        return;
    }

    // Check if the output dimensions match the expected sizes
    if ((H - R + 2 * padding) / stride + 1 != conved_H || (W - S + 2 * padding) / stride + 1 != conved_W)
    {
        printf("Err: not paired conv src and result space\n");
        return;
    }
    cudnnHandle_t cudnn;
    CHECK_CUDNN_ERR(cudnnCreate(&cudnn));

    // Allocate device memory for input, output, and kernels
    float *d_input, *d_output, *d_kernels;
    cudaMalloc((void **)&d_input, C * H * W * sizeof(float));
    cudaMalloc((void **)&d_output, h_output.matrix_nums * conved_H * conved_W * sizeof(float));
    cudaMalloc((void **)&d_kernels, kernelSet.numKernels * C * R * S * sizeof(float));

    // Copy input and kernel data from host to device
    cudaMemcpy(d_input, h_input.h_input, C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernelSet.deviceKernels, kernelSet.numKernels * C * R * S * sizeof(float), cudaMemcpyHostToDevice);

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    // Create cuDNN descriptors
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&kernel_desc));
    CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input tensor descriptor (NCHW format)
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    // Set output tensor descriptor (NCHW format)
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, h_output.matrix_nums, conved_H, conved_W));

    // Set kernel descriptor (filter)
    CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N, C, R, S));

    // Set convolution descriptor
    CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Get the output dimensions from cuDNN
    int n, c, h, w;
    CHECK_CUDNN_ERR(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, kernel_desc, &n, &c, &h, &w));

    // Check the computed dimensions
    if (n != 1 || c != h_output.matrix_nums || h != conved_H || w != conved_W)
    {
        printf("expected: %d,%d,%d\n", h_output.matrix_nums, conved_H, conved_W);
        printf("Err: computed output dimensions don't match expected ones. Got (%d, %d, %d, %d)\n", n, c, h, w);
        return;
    }

    // Set the output tensor descriptor with the computed dimensions
    CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, c, h, w));

    // Perform the convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN_ERR(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernels, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, output_desc, d_output));

    // Copy the result back to host memory
    cudaMemcpy(h_output.h_input, d_output, h_output.matrix_nums * conved_H * conved_W * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory and cuDNN descriptors
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernels);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}