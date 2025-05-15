#ifndef CLASS_H
#define CLASS_H
#include "utilis.cuh"
#include <iostream>
#include <string>
#include <cuda_runtime.h>

// matrixs �ṹ�壺���ڴ洢�Ͳ����������
// ������������������д�С�����������Լ���ز���

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

// Kernel �ࣺ���ڴ洢�Ͳ��������
// ��������˵�������ÿ������˵Ĵ�С�����豸�ڴ�ָ��
class Kernel {
public:
    int numKernels;             // ��������
    int rows;                   // ÿ�����ĵ�����
    int cols;                   // ÿ�����ĵ�����
    const float *deviceKernels; // �洢�� GPU �ϵĺ������ݣ����Ϊ const ʹ�䲻���޸�

    // ���캯������ʼ���������������������������豸�ڴ�
    Kernel(int numKernels, int rows, int cols);

    // �����������ͷ��豸�ڴ�
    ~Kernel();

    // �������ڴ��е����ݸ��Ƶ��豸�ڴ�
    void setKernelValueHost(int kernelIndex, float *h_kernel);

    // ��ӡ�豸�Ϻ������ݵ�ֵ
    void printKernelValues(int kernelIndex) const;

    // ��ӡ���к��ĵ�ά����Ϣ��rows, cols��
    void printKernelDimensions() const;

    // ��ȡ�豸�ڴ��еĺ������ݣ�ֻ����
    const float* getDeviceKernels() const;
};

#endif // KERNEL_H
