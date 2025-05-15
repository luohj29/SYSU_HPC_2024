#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/conv.cuh"
#include "../include/utilis.cuh"

/*
    input: m,n for matrix size; kernel_nums, kernel_shape in(a, b), padding stride
*/

int main(int argc, char *argv[])
{
    std::cout << "argc: " << argc << std::endl;
    if (argc != 8)
    {
        printf("not enough params\n");
        return 0;
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int kernel_nums = atoi(argv[3]);
    int kernel_m = atoi(argv[4]);
    int kernel_n = atoi(argv[5]);
    int padding = atoi(argv[6]);
    int stride = atoi(argv[7]);

    int padded_m = m + 2 * padding;
    int padded_n = n + 2 * padding;
    int conved_m = (m - kernel_m + 2 * padding) / stride + 1;
    int conved_n = (n - kernel_n + 2 * padding) / stride + 1;

    float *h_input;
    cudaMallocHost((void **)&h_input, sizeof(float) * m * n);

    randomize_matrix_simply(h_input, m, n);             // 初始化输入矩阵
    Kernel kernel_set(kernel_nums, kernel_m, kernel_n); // 创建卷积核
    Kernel kernel_set2(kernel_nums, kernel_m, kernel_n);
    float *kernel_mod;
    cudaMallocHost((void **)&kernel_mod, kernel_m * kernel_n * sizeof(float));
    randomize_matrix_simply(kernel_mod, kernel_m, kernel_n);
    matrixs matrixs_input(3, m, n);
    matrixs matrixs_output(1, conved_m, conved_n);

    for (int i = 0; i < kernel_nums; i++)
    {
        kernel_set.setKernelValueHost(i, kernel_mod);
        kernel_set2.setKernelValueHost(i, kernel_mod);
        matrixs_input.set_stacked_matrix(i, h_input);
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((padded_m + BLOCK_SIZE - 1) / BLOCK_SIZE, (padded_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("blockx: %d blocky: %d gridx: %d gridy:%d\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------- 普通滑动窗口卷积 --------------------------------
    cudaEventRecord(start);
    conv2d(dimBlock, dimGrid, matrixs_input, matrixs_output, kernel_set, stride, padding);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sliding_conv_time = 0;
    cudaEventElapsedTime(&sliding_conv_time, start, stop);
    printf_matrix_corner(matrixs_output.h_input, matrixs_output.rows, matrixs_output.cols);
    printf("Custom Kernel Time for sliding conv: %f ms\n\n", sliding_conv_time);

    // -------------------------------- Img2col 卷积 --------------------------------
    cudaEventRecord(start);
    conv2d_gemm_v2(dimBlock, dimGrid, matrixs_input, matrixs_output, kernel_set2, stride, padding);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float img2col_conv_time = 0;
    cudaEventElapsedTime(&img2col_conv_time, start, stop);
    printf_matrix_corner(matrixs_output.h_input, matrixs_output.rows, matrixs_output.cols);
    printf("Custom Kernel Time for img2colv2 conv: %f ms\n", img2col_conv_time);

    // --------------------------------cudnn卷积--------------------------------------------
    cudaEventRecord(start);
    conv_cudnn(matrixs_input, matrixs_output, kernel_set2, stride, padding);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cudnn_conv_time = 0;
    cudaEventElapsedTime(&cudnn_conv_time, start, stop);
    printf_matrix_corner(matrixs_output.h_input, matrixs_output.rows, matrixs_output.cols);
    printf("Custom Kernel Time for cudnn conv: %f ms\n", cudnn_conv_time);
    // 释放资源
    cudaFree(h_input);
    cudaFree(kernel_mod);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}