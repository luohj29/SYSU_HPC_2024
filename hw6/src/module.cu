// #include"module.h"
#ifndef MODULE_H
#define MODULE_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <string>
#include "cuda_runtime.h"
#include <sys/time.h>
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
#define BLOCK_SIZE 32

void gpu_gemm_cublas(const float *d_a, const float *d_b, float *d_result, int m, int n, int k)
{
    //   m*n x n*k =m *k
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Configure cuBLAS operations
    const float alpha = 1.0, beta = 0.0;
    cublasOperation_t opA = CUBLAS_OP_N;                                           // No transpose for A
    cublasOperation_t opB = CUBLAS_OP_N;                                           // No transpose for B
    cublasSgemm(handle, opA, opB, k, m, n, &alpha, d_b, k, d_a, n, &beta, d_result, k); 
    // cudaMemcpy(h_output.h_input, d_result, sizeof(float)*converted_n, cudaMemcpyDeviceToHost);
}



void printf_matrix_corner(float *data, int m, int n)
{
    if (m < 2 || n < 2)
    {
        printf("Matrix dimensions are too small to display corners.\n");
        return;
    }

    printf("%.4f %.4f ... %.4f %.4f\n",
           data[0], data[1], data[n - 2], data[n - 1]);

    printf("%.4f ...... ... ...... %.4f\n",
           data[n], data[2 * n - 1]);
    printf("...... ...... ... ...... ......\n");

    printf("%.4f ...... ... ...... %.4f\n",
           data[(m - 2) * n], data[(m - 2) * n + n - 1]);

    printf("%.4f %.4f ... %.4f %.4f\n",
           data[(m - 1) * n], data[(m - 1) * n + 1], data[(m - 1) * n + n - 2], data[(m - 1) * n + n - 1]);
}

// Randomize the values of a matrix
void randomize_matrix(float *mat, int m, int n)
{
    int N = m * n;
    struct timeval time;
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++)
    {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.0f);
        mat[i] = tmp;
    }
}

// A simplified version of randomize_matrix
void randomize_matrix_simply(float *mat, int m, int n)
{
    int N = m * n;
    for (int i = 0; i < N; i++)
    {
        mat[i] = (i + 1) % 10;
    }
}

// Get and print device information (GPU)
void get_device_info()
{
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "  Clock Rate (GHz): " << prop.clockRate / 1e6 << " GHz" << std::endl;
    std::cout << "  CUDA Cores: " << prop.multiProcessorCount * 128 << std::endl;
    std::cout << "  Theoretical Peak Performance (FP32): "
              << prop.multiProcessorCount * 128 * 2 * (prop.clockRate / 1e6) / 1e3
              << " TFLOPS" << std::endl;
    printf("Shared memory per block:  %zu  bytes\n", prop.sharedMemPerBlock);
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid size: (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max block size (x, y, z): ("
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << std::endl;
}


/**
 * @brief  implemenntation of the GEMM cuda in global memory
 * 
 * @param a  the first matrix in m*n in device memory
 * @param b  the second matrix in n*k in device memory
 * @param c  the result matrix in m*k in device memory
 * @param m  matrix size
 * @param n  matrix size
 * @param k  matrix size 
 * @return the result in ptr c
 */
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
        // printf("%d %d index:%d, reuslt: %d\n", row, col, row * k + col ,temp);
    }
}

/**
 * @brief  implemenntation of the GEMM cuda in shared memory
 *
 * @param a  the first matrix in m*n in device memory
 * @param b  the second matrix in n*k in device memory
 * @param c  the result matrix in m*k in device memory
 * @param m  matrix size
 * @param n  matrix size
 * @param k  matrix size
 * @return the result in ptr c
 */
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
            tile_a[threadIdx.x][threadIdx.y] = a[a_index];
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
            temp += tile_a[j][threadIdx.y] * tile_b[j][threadIdx.x];
        }
        __syncthreads(); // Ensure all threads finish their computation before moving on
    }
    if (row < m && col < k)
    {
        c[row * k + col] = temp;
    }
}

__global__ void gpu_matrix_mult_sm_v2(float *a, float *b, float *c, int m, int n, int k)
{
    const int BM = 128;
    const int BN = 8;
    const int BK =128;
    const int TM = 8;
    const int TN = 8;
    const int Row  = blockIdx.y;
    const int Col = blockIdx.x;
    // blocktile the original matrix into shared memory
    __shared__ float As[BM][BN];
    __shared__ float Bs[BN][BK];

    // switch to the bagining of the blocktile for each thread
    a += Row * BM * n;  //down the row
    b += Col * BK; //down the col
    c += Row * BM * n + Col * BK; //switch to the corresponding block in c


}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

__global__ void sgemm2DWarpTiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const uint strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
        {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
        {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // block into registers
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
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
    {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
}
/**
 * @brief  implemenntation of the GEMM cuda in cpu memory
 *
 * @param a  the first matrix in m*n in host memory
 * @param b  the second matrix in n*k in host memory
 * @param c  the result matrix in m*k in host memory
 * @param m  matrix size
 * @param n  matrix size
 * @param k  matrix size
 * @return the result in ptr h_result
 */
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

/**
 * @brief print a matrix all with a text for remindation
 * 
 * @param s  the text for reminding what the matrix is
 * @param data  the matrix ptr
 * @param m  row  dim
 * @param n  col dim
 */
void print_all_matrix(std::string s, float *data, int m, int n)
{
    std::cout << "Printing matrix of " << s << std::endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", data[i * n + j]);
        }
        printf("\n");
    }
}

#include <iostream>
#include <cuda_runtime.h>

#include <iostream>
#include <cuda_runtime.h>

class Kernel
{
public:
    int numKernels;             
    int rows;                   
    int cols;                   
    const float *deviceKernels; 

 
    Kernel(int numKernels, int rows, int cols)
        : numKernels(numKernels), rows(rows), cols(cols), deviceKernels(nullptr)
    {
        cudaError_t err = cudaMalloc((void **)&deviceKernels, numKernels * rows * cols * sizeof(float));

        if (err != cudaSuccess)
        {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        }
    }


    ~Kernel()
    {
        cudaFree(const_cast<float *>(deviceKernels)); 
    }


    void setKernelValueHost(int kernelIndex, float *h_kernel)
    {
        cudaMemcpy((void *)(deviceKernels + kernelIndex * rows * cols), h_kernel, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    }

 
    void printKernelValues(int kernelIndex) const
    {
        float *h_kernel = new float[rows * cols]; 


        cudaMemcpy(h_kernel, deviceKernels + kernelIndex * rows * cols, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);


        std::cout << "Kernel " << kernelIndex << " values:\n";
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                std::cout << h_kernel[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }

        delete[] h_kernel; 
    }

  
    void printKernelDimensions() const
    {
        std::cout << "Kernel dimensions: " << rows << " x " << cols << std::endl;
        std::cout << "Number of kernels: " << numKernels << std::endl;
    }

#include <assert.h>
    const float *getDeviceKernels() const
    {
        return deviceKernels;
    }
};

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
                    // if (row == 0 && col == 3 && k == 0)
                    //     //     // printf("In kernel curve j:%d sum:%d\n", j, kernel_idx);
                    //     printf("row: %d, col: %d, i: %d, j:%d, input:%f, kernel:%f, temp: %f\n", row, col, i, j, input[input_row * paddedn + input_col], kernelSet.deviceKernels[kernel_idx], temp);

                    // if (row == 0 && col == 0)
                    //     printf("i: %f, j:%f, input:%f, kernel:%f, temp: %f\n", i, j, input[input_row * paddedn + input_col],
                    //            kernelSet.deviceKernels[kernel_idx], temp[k]);
                }
            }
            // 写入输出           
        }
        output[(row)/stride * conved_n + (col)/stride] = temp;
    }
}

class matrixs
{
public:
    int matrix_nums;
    int rows;        
    int cols;        
    float *h_input;  


    matrixs(int matrix_nums, int rows, int cols)
        : matrix_nums(matrix_nums), rows(rows), cols(cols), h_input(nullptr)
    {

        h_input = (float *)malloc(sizeof(float) * matrix_nums * rows * cols); 


        std::fill(h_input, h_input + matrix_nums * rows * cols, 0.0f);
    }


    ~matrixs()
    {
    }

    void delete_manually()
    {
        if (h_input != NULL)
        {
            free(h_input);
            h_input = NULL; 
        }
    }
    void set_stacked_matrix(int index, float *input)
    {
        if (index >= matrix_nums)
        {
            printf("Err index!!!\n");
            return;
        }
        int layers = rows * cols;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                h_input[index * layers + i * cols + j] = input[i * cols + j];
            }
        }
        return;
    }
    void print_stacked_matrix()
    {
        for (int k = 0; k < matrix_nums; k++)
        {
            std::string s = "printing " + std::to_string(k) + "th matrix";
            print_all_matrix(s, &h_input[k * rows * cols], rows, cols);
            printf("\n");
        }
    }
};

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

#define CHECK_CUDA_CALL(call)                                                \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err);          \
            std::cerr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }
/*
    description:
    - input and output are both in host mem
    - make the device mem, use the con2d_cal to finish the computation part
*/
#include <string.h>
void conv2d(dim3 Block, dim3 Grid, matrixs input, matrixs output,   Kernel kernelSet, int stride, int padding)
{
    
    // printf("hello\n");
    float *d_input;
    float *h_input = input.h_input;
    float *h_output = output.h_input;
    int m = input.rows;
    int n = input.cols;
    int paddedRows = m + 2 * padding;
    int paddedCols = n + 2 * padding;
    float *temp;
    int x = input.matrix_nums;

    assert(input.matrix_nums == kernelSet.numKernels);  //channel should be the same
    cudaMallocHost((void **)&temp, sizeof(float) * paddedRows * paddedCols * input.matrix_nums);

    cudaMalloc((void **)&d_input, sizeof(float) * paddedRows * paddedCols * x);

    memset(temp, 0, sizeof(float) * paddedRows * paddedCols * input.matrix_nums);  //需要把原来的内存设置为0，否则padding的时候可能会出错

    padMatrix(h_input, temp, m, n, input.matrix_nums, padding);
    
    // print_all_matrix("filter: ", kernelSet.deviceKernels, 3, 3);
    // print_all_matrix("temp: ", temp, paddedRows, paddedCols);
    // kernelSet.printKernelValues(0);
    cudaMemcpy(d_input, temp, sizeof(float) * paddedRows * paddedCols * input.matrix_nums, cudaMemcpyHostToDevice);


    cudaMalloc((void **)&temp, sizeof(float) * kernelSet.rows * kernelSet.cols* input.matrix_nums);
    cudaMemcpy(temp, kernelSet.deviceKernels, sizeof(float) * kernelSet.rows * kernelSet.cols * input.matrix_nums, cudaMemcpyDeviceToDevice);  //protect
    float *d_output;
    int conved_m = (m - kernelSet.rows + 2 * padding) / stride + 1;
    int conved_n = (n - kernelSet.cols + 2 * padding) / stride + 1;
    int output_block = conved_m * conved_n; // for indexing erery kernel output!
    cudaMalloc((void **)&d_output, sizeof(float)  * output_block);
    // printf("before\n");

    // kernelSet.printKernelValues(0);

    conv2d_cal<<<Grid, Block>>>(d_input, d_output, output_block, m, n, kernelSet, stride, padding); // into the device domain

    // kernelSet.printKernelValues(0);

    cudaMemcpy(h_output, d_output, sizeof(float) * output_block, cudaMemcpyDeviceToHost);
    // print_all_matrix("output: ", h_output, conved_m, conved_n);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(temp);
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

cudaPitchedPtr allocate3DDeviceMemory(size_t width, size_t height, size_t depth)
{

    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);


    cudaPitchedPtr d_matrix;


    cudaError_t err = cudaMalloc3D(&d_matrix, extent);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc3D failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }


    return d_matrix;
}

enum CopyType
{
    HostToDevice = 0, 
    DeviceToHost = 1  
};

void copy1DTo3DDeviceMemory(void *host_data, void *device_data, size_t width, size_t height, size_t depth, CopyType cpy_type)
{
    size_t pitch = width * sizeof(float); 

    if (cpy_type == HostToDevice)
    {

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(host_data, pitch, width, height);
        copyParams.dstPtr = make_cudaPitchedPtr(device_data, pitch, width, height);
        copyParams.extent = make_cudaExtent(width * sizeof(float), height, depth);
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
    }
    else if (cpy_type == DeviceToHost)
    {

        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(device_data, pitch, width, height);
        copyParams.dstPtr = make_cudaPitchedPtr(host_data, pitch, width, height);
        copyParams.extent = make_cudaExtent(width * sizeof(float), height, depth);
        copyParams.kind = cudaMemcpyDeviceToHost;
        cudaMemcpy3D(&copyParams);
    }
    else
    {
        std::cerr << "Invalid copy type!" << std::endl;
    }
}
// __global__ void gpu_matrix_mult_gm(float *a, float *b, float *c, int m, int n, int k)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float temp = 0;
//     if (row < m && col < k) // Ensure bounds are within the matrix dimensions
//     {
//         for (int i = 0; i < n; i++)
//         {
//             temp += a[row * n + i] * b[i * k + col];
//         }
//         c[row * k + col] = temp;
//     }
// }

__global__ void conv2d_gemm_cal(cudaPitchedPtr d_matrix, cudaPitchedPtr d_filter, cudaPitchedPtr d_result, int filter_size, int d_matrix_n, int depth)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;


    if (y < 1 && x < d_matrix_n && z < depth) // 1*filter_size x filter_size * d_matrix_n
    {

        float *a = (float *)d_filter.ptr; // filter matrix
        float *b = (float *)d_matrix.ptr; // input matrix
        float *c = (float *)d_result.ptr; // output matrix
        int a_layer = 1 * filter_size;
        int b_layer = filter_size * d_matrix_n;
        int c_layer = 1 * d_matrix_n;

        for (int k = 0; k < depth; k++)
        { 
            float temp = 0.0;
            for (int i = 0; i < filter_size; i++)
            { 
                temp += a[k * a_layer + i + y] * b[k * b_layer + i * d_matrix_n + x];
            }
            c[k * c_layer + x] = temp;
        }
    }
}


/*
    descroption: this function is about using img2col alg to do convolution
*/
void conv2d_gemm(dim3 Block, dim3 Grid, matrixs h_input, matrixs h_output, Kernel kernelSet, int stride, int padding)
{
    // printf("hello\n");
    int padded_m = h_input.rows + 2 * padding;
    int padded_n = h_input.cols + 2 * padding;
    int convable_m = (h_input.rows + 2 * padding - kernelSet.rows) / stride + 1;
    int convable_n = (h_input.cols + 2 * padding - kernelSet.cols) / stride + 1;

    int converted_n = convable_m * convable_n;
    int converted_m = kernelSet.rows * kernelSet.cols;

    int block = converted_m * converted_n;
    int padded_block = padded_m * padded_n;


    float *temp;
    // printf("hello\n");
    cudaMallocHost((void **)&temp, sizeof(float) * padded_block * h_input.matrix_nums);  //create a temp to get the padded matrix in host mem

    padMatrix(h_input.h_input, temp, h_input.rows, h_input.cols, h_input.matrix_nums, padding);


    float *h_temp;
    cudaMallocHost((void **)&h_temp, sizeof(float) * h_input.matrix_nums * block);  //for converted matrix
    for (int i = 0; i +kernelSet.rows-1 < padded_m; i+=stride) // the next 2 iterations are for sliding the conv window
    {
        for (int j = 0;  j+kernelSet.cols-1<padded_n; j+=stride)
        {
            for (int a = 0; a < kernelSet.rows; a++) // the naxt 2 iterations are for inside the conv window
            {
                for (int b = 0; b < kernelSet.cols; b++)
                {
                    for (int k = 0; k < h_input.matrix_nums; k++) // the k iteration for the input matrixs(could be more than 1)
                    {
                        int li = k * block + (a * kernelSet.cols + b) * converted_n + (i/stride) * convable_n + (j/stride);
                        int ri = k * padded_block + (i + a) * padded_m + j + b;
                        h_temp[li] = temp[ri];
                        // if(i ==0 &&j ==0 && k==0)
                        //     printf("li:%d, ri%d, %f %f\n", li,ri, h_temp[li], temp[ri]);
                    }
                }
            }
        }
    }
    // print_all_matrix("filter: ", kernelSet.deviceKernels, kernelSet.rows, kernelSet.cols);

    // print_all_matrix("converted:", h_temp, converted_m, converted_n);
    
    cudaFreeHost(temp);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
    //     return; 
    // }

    // auto d_matrix = allocate3DDeviceMemory(converted_n, converted_m, h_input.matrix_nums);
    float *d_matrix, *d_filter, *d_result;
    cudaMalloc((void **)&d_matrix, converted_n*converted_m*h_input.matrix_nums*sizeof(float));
    cudaMemcpy(d_matrix, h_temp, converted_n * converted_m * h_input.matrix_nums * sizeof(float), cudaMemcpyHostToDevice);
    // copy1DTo3DDeviceMemory((void *)h_temp, (void *)d_matrix.ptr, converted_n, converted_m, h_input.matrix_nums, HostToDevice);

    cudaFreeHost(h_temp);

    int filter_size = kernelSet.cols * kernelSet.rows;
    // auto d_filter = allocate3DDeviceMemory(filter_size, 1, kernelSet.numKernels);
    cudaMallocHost((void **)&d_filter, filter_size * h_input.matrix_nums * sizeof(float));
    
    cudaMemcpy(d_filter, kernelSet.deviceKernels, filter_size * h_input.matrix_nums * sizeof(float), cudaMemcpyDeviceToHost);
    // print_all_matrix("filter: ", d_filter, 1, kernelSet.numKernels * filter_size);
    // kernelSet.printKernelValues(2);
    // copy1DTo3DDeviceMemory((void *)kernelSet.deviceKernels, (void *)d_filter.ptr, kernelSet.cols * kernelSet.rows, 1, kernelSet.numKernels, HostToDevice);

    int blockx = 512, blocky = 1;
    dim3 blockDim(blockx, blocky);
    int gridx = (converted_n + blockx - 1) / blockx;
    int gridy = 1;
    dim3 gridDim(gridx, gridy);

    // auto d_result = allocate3DDeviceMemory(converted_n, 1, h_input.matrix_nums);
    cudaMalloc((void **)&d_result, sizeof(float) * converted_n);
    // conv2d_gemm_cal<<<gridDim, blockDim>>>(d_matrix, d_filter, d_result, filter_size, converted_n, kernelSet.numKernels);
    gpu_matrix_mult_gm<<<gridDim, blockDim>>>(kernelSet.deviceKernels, d_matrix, d_result, 1, filter_size * kernelSet.numKernels, converted_n);
    //   m*k k*n = m*n
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // // Configure cuBLAS operations
    // const float alpha = 1.0, beta = 0.0;
    // cublasOperation_t opA = CUBLAS_OP_N;                                           // No transpose for A
    // cublasOperation_t opB = CUBLAS_OP_N;                                           // No transpose for B
    // cublasSgemm(handle, opA, opB, converted_n, 1, filter_size*3, &alpha, d_matrix, converted_n, kernelSet.deviceKernels, filter_size * 3, &beta, d_result, converted_n); // 我日，cublas 居然是列优先的
    gpu_gemm_cublas(kernelSet.deviceKernels, d_matrix, d_result, 1, filter_size * kernelSet.numKernels, converted_n);
    cudaMemcpy(h_output.h_input, d_result, sizeof(float) * converted_n, cudaMemcpyDeviceToHost);
    // print_all_matrix("h_ouput: ", h_output.h_input, convable_m, convable_n);
    // copy1DTo3DDeviceMemory((void *)h_output.h_input, (void *)d_result.ptr, convable_n, convable_m, kernelSet.numKernels, DeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_filter);
    cudaFree(d_result);

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
    
*/
// void conv_cudnn(float* h_input, float* h_output, float* kernelSet, int rows ,int cols, int channels, int kernel_row, int kernel_col, int stride, int padding)
// {
//     int N = 1; // Batch size
//     int C = h_input.matrix_nums;
//     int H = h_input.rows;
//     int W = h_input.cols;

//     int R = kernelSet.rows;
//     int S = kernelSet.cols;

//     int conved_H = h_output.rows;
//     int conved_W = h_output.cols;

//     // Check if the kernel count matches the input channels
//     if (kernelSet.numKernels != h_input.matrix_nums)
//     {
//         printf("Err: not paired kernels and input\n");
//         return;
//     }

//     // Check if the output dimensions match the expected sizes
//     if ((H - R + 2 * padding) / stride + 1 != conved_H || (W - S + 2 * padding) / stride + 1 != conved_W)
//     {
//         printf("Err: not paired conv src and result space\n");
//         return;
//     }

//     cudnnHandle_t cudnn;
//     CHECK_CUDNN_ERR(cudnnCreate(&cudnn));

//     // Allocate device memory for input, output, and kernels
//     float *d_input, *d_output, *d_kernels;
//     cudaMalloc((void **)&d_input, C * H * W * sizeof(float));
//     cudaMalloc((void **)&d_output, h_output.matrix_nums * conved_H * conved_W * sizeof(float));
//     cudaMalloc((void **)&d_kernels, kernelSet.numKernels * C * R * S * sizeof(float));

//     // Copy input and kernel data from host to device
//     cudaMemcpy(d_input, h_input.h_input, C * H * W * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_kernels, kernelSet.deviceKernels, kernelSet.numKernels * C * R * S * sizeof(float), cudaMemcpyHostToDevice);

//     cudnnTensorDescriptor_t input_desc, output_desc;
//     cudnnFilterDescriptor_t kernel_desc;
//     cudnnConvolutionDescriptor_t conv_desc;

//     // Create cuDNN descriptors
//     CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&input_desc));
//     CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&output_desc));
//     CHECK_CUDNN_ERR(cudnnCreateFilterDescriptor(&kernel_desc));
//     CHECK_CUDNN_ERR(cudnnCreateConvolutionDescriptor(&conv_desc));

//     // Set input tensor descriptor (NCHW format)
//     CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

//     // Set output tensor descriptor (NCHW format)
//     CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, h_output.matrix_nums, conved_H, conved_W));

//     // Set kernel descriptor (filter)
//     CHECK_CUDNN_ERR(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kernelSet.numKernels, C, R, S));

//     // Set convolution descriptor
//     CHECK_CUDNN_ERR(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

//     // Get the output dimensions from cuDNN
//     int n, c, h, w;
//     CHECK_CUDNN_ERR(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, kernel_desc, &n, &c, &h, &w));

//     // Check the computed dimensions
//     if (n != 1 || c != h_output.matrix_nums || h != conved_H || w != conved_W)
//     {
//         printf("Err: computed output dimensions don't match expected ones. Got (%d, %d, %d, %d)\n", n, c, h, w);
//         return;
//     }

//     // Set the output tensor descriptor with the computed dimensions
//     CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, c, h, w));

//     // Perform the convolution
//     float alpha = 1.0f, beta = 0.0f;
//     CHECK_CUDNN_ERR(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernels, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, output_desc, d_output));

//     // Copy the result back to host memory
//     cudaMemcpy(h_output., d_output, h_output.matrix_nums * conved_H * conved_W * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory and cuDNN descriptors
//     cudaFree(d_input);
//     cudaFree(d_output);
//     cudaFree(d_kernels);
//     cudnnDestroyTensorDescriptor(input_desc);
//     cudnnDestroyTensorDescriptor(output_desc);
//     cudnnDestroyFilterDescriptor(kernel_desc);
//     cudnnDestroyConvolutionDescriptor(conv_desc);
//     cudnnDestroy(cudnn);
// }
#endif