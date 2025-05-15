#include "../include/class.cuh"

matrixs::matrixs(int matrix_nums, int rows, int cols)
    : matrix_nums(matrix_nums), rows(rows), cols(cols), h_input(nullptr)
{
    h_input = (float *)malloc(sizeof(float) * matrix_nums * rows * cols);
    std::fill(h_input, h_input + matrix_nums * rows * cols, 0.0f);
}

matrixs::~matrixs()
{
}

void matrixs::delete_manually()
{
    if (h_input != nullptr)
    {
        free(h_input);
        h_input = nullptr;
    }
}

void matrixs::set_stacked_matrix(int index, float *input)
{
    if (index >= matrix_nums)
    {
        std::cerr << "Err index!!!" << std::endl;
        return;
    }
    int layers = rows * cols;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            h_input[index * layers + i * cols + j] = input[i * cols + j];
        }
    }
}

void matrixs::print_stacked_matrix()
{
    for (int k = 0; k < matrix_nums; ++k)
    {
        std::string s = "printing " + std::to_string(k) + "th matrix";
        print_all_matrix(s, &h_input[k * rows * cols], rows, cols);
        std::cout << std::endl;
    }
}

void matrixs::print_all_matrix(const std::string &title, const float *matrix, int rows, int cols) const
{
    std::cout << title << ":\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


Kernel::Kernel(int numKernels, int rows, int cols)
    : numKernels(numKernels), rows(rows), cols(cols), deviceKernels(nullptr)
{
    cudaError_t err = cudaMalloc((void **)&deviceKernels, numKernels * rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    }
}

Kernel::~Kernel()
{
    cudaFree(const_cast<float *>(deviceKernels));
}

void Kernel::setKernelValueHost(int kernelIndex, float *h_kernel)
{
    cudaMemcpy((void *)(deviceKernels + kernelIndex * rows * cols), h_kernel, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

void Kernel::printKernelValues(int kernelIndex) const
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

void Kernel::printKernelDimensions() const
{
    std::cout << "Kernel dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Number of kernels: " << numKernels << std::endl;
}

const float *Kernel::getDeviceKernels() const
{
    return deviceKernels;
}
