#include "../include/utilis.cuh"

// Floating point comparison with epsilon tolerance
bool is_data_eq(float x, float y)
{
    return x == y || (abs(x - y) < EPISILON);
}

// Print the corners of a matrix
void printf_matrix_corner(float *data, int m, int n)
{
    if (m < 2 || n < 2)
    {
        printf("Matrix dimensions are too small to display corners.\n");
        return;
    }

    printf("%.4f %.4f ... %.4f %.4f\n",
           data[0], data[1], data[n - 2], data[n - 1]);

    printf("%.4f %.4f ... %.4f %.4f\n",
           data[n], data[n + 1], data[2 * n - 2], data[2 * n - 1]);
    printf("...... ...... ... ...... ......\n");

    printf("%.4f %.4f ... %.4f %.4f\n",
           data[(m - 2) * n], data[(m - 2) * n + 1], data[(m - 2) * n + n - 2], data[(m - 2) * n + n - 1]);

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
}

// Print a matrix with a description string
void print_all_matrix(std::string s, float *data, int m, int n)
{
    std::cout << "Printing matrix of " << s << std::endl;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.4f ", data[i * n + j]);
        }
        printf("\n");
    }
}
