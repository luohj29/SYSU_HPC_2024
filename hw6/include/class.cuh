#ifndef CLASS_H
#define CLASS_H
#include "utilis.cuh"
#include <iostream>
#include <string>
#include <cuda_runtime.h>

// matrixs 结构体：用于存储和操作多个矩阵
// 包含矩阵的数量、行列大小、矩阵数据以及相关操作

class matrixs
{
public:
    int matrix_nums;
    int rows;
    int cols;
    float *h_input;

    matrixs(int matrix_nums, int rows, int cols);
    ~matrixs();
    void delete_manually();
    void set_stacked_matrix(int index, float *input);
    void print_stacked_matrix();

private:
    void print_all_matrix(const std::string &title, const float *matrix, int rows, int cols) const;
};

// Kernel 类：用于存储和操作卷积核
// 包含卷积核的数量、每个卷积核的大小及其设备内存指针
class Kernel {
public:
    int numKernels;             // 核心数量
    int rows;                   // 每个核心的行数
    int cols;                   // 每个核心的列数
    const float *deviceKernels; // 存储在 GPU 上的核心数据，标记为 const 使其不可修改

    // 构造函数：初始化核心数量、行列数，并分配设备内存
    Kernel(int numKernels, int rows, int cols);

    // 析构函数：释放设备内存
    ~Kernel();

    // 将主机内存中的数据复制到设备内存
    void setKernelValueHost(int kernelIndex, float *h_kernel);

    // 打印设备上核心数据的值
    void printKernelValues(int kernelIndex) const;

    // 打印所有核心的维度信息（rows, cols）
    void printKernelDimensions() const;

    // 获取设备内存中的核心数据（只读）
    const float* getDeviceKernels() const;
};

#endif // KERNEL_H
