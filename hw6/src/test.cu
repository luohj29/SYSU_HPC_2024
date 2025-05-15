#include <iostream>
#include <cuda_runtime.h>

int main()
{
    // 获取 GPU 设备数量
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }

    // 遍历所有 GPU 设备
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Threads per Warp: " << deviceProp.warpSize << std::endl;

        // 计算 Max Warps per Multiprocessor
        int maxWarpsPerMP = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
        std::cout << "  Max Warps per Multiprocessor: " << maxWarpsPerMP << std::endl;

        std::cout << "  Max Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Max Registers per Multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;

        // 计算 Registers Allocation Granularity
        // 通常寄存器的分配粒度是 Warp（32 个线程）
        std::cout << "  Registers Allocation Granularity: Warp" << std::endl;

        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Max Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory per Multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << std::endl;
    }

    return 0;
}