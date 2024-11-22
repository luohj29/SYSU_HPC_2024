#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void matrix_multiply_s(int M, int N, int K, float* A, float* B, float* C) {
/*
    输入三个矩阵的维度，以及三个矩阵的二维矩阵，输出计算时间
*/
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i*M+j] = 0;
        }
    }

    // clock_t start, end;
    // start = clock();

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < N; k++){
            for (int j = 0; j < K; j++){  
                C[i*K+j] += A[i*N+k] * B[k*K+j];
            }
        }
    }

    // end = clock();

    // printf("C Matrix Multiplying in sequence of %d*%d and %d*%d use time of %f seconds\n", M, N, N, K, (double)(end - start) / CLOCKS_PER_SEC);

    return ;
}

void printmatrix(int M, int N, float *A){
    for(int i=0;i<M;i++){
        for(int j=0; j<N;j++){
            printf("%f ", A[i*N+j]);
        }
        printf("\n");
    }
}