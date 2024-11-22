#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include "parallel.h"

int THREAD_NUMS, SIZE;

void * matrix_multiply_s(void * args) {
    struct for_index * index = (struct for_index *) args;
    int N=SIZE,K =SIZE;
    // clock_t start, end;
    // start = clock();
    float* A=index->A;
    float* B=index->B;
    float* C=index->C;
    printf("%d %d\n", index->start,index->end); 
    for (int i = index->start; i < index->end; i++) {

        for (int k = 0; k < N; k++){
            for (int j = 0; j < K; j++){ 
                C[i*K+j] += A[i*N+k] * B[k*K+j];
            }
        }
    }

    // end = clock();

    // printf("C Matrix Multiplying in sequence of %d*%d and %d*%d use time of %f seconds\n", M, N, N, K, (double)(end - start) / CLOCKS_PER_SEC);

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("NEED 3 PARA!!!\n");
        return 1;
    }
    THREAD_NUMS = atoi(argv[1]);
    SIZE = atoi(argv[2]); 
    // int mode =  atoi(argv[3]);

    srand((unsigned)time(NULL));	//使用当前系统时间作为种子
    // AVG_NUM = SIZE/THREAD_NUMS;
    int M=SIZE;
    int N=SIZE;
    int K=SIZE;
    float *A = (float*)malloc(sizeof(float)*M*N);
    float *B = (float*)malloc(sizeof(float)*N*K);
    float *C = (float*)malloc(sizeof(float)*M*K);
    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // A[i * N + j] = (float)rand() / RAND_MAX * 10000.0;
            A[i * N + j] =3;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            // B[i * K + j] = (float)rand() / RAND_MAX * 10000.0;
            B[i * K + j]=3;
        }
    }
     for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i * K+ j] = 0;
        }
    }
   
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    // Parallel matrix multiplication  
    parallel_for(A,B,C ,0, M, 0, matrix_multiply_s, NULL, THREAD_NUMS);
    gettimeofday(&end, NULL);

    double time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    time_use /= 1000000;
    for(int j=K-10; j<K;j++){
        printf("%.4f ", C[(K-1)*K +j]);
    }
    printf("\nReal time used: %.4f seconds\n", time_use);
    free(A);
    free(B);
    free(C);
    return 0;
}