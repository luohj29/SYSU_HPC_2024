#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/*
    输入三个矩阵的维度，以及三个矩阵的二维矩阵，输出计算时间
*/

extern void matrix_multiply_s(int M, int N, int K, float* A, float* B, float* C);
