#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

int THREAD_NUMS, SIZE, AVG_NUM;
int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("NEED 3 PARA!!!\n");
        return 1;
    }
    THREAD_NUMS = atoi(argv[1]);
    SIZE = atoi(argv[2]); 
    int mode =  atoi(argv[3]);

    AVG_NUM = SIZE/THREAD_NUMS;
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
            A[i * N + j] = 3;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i * K + j] = 3;
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

    switch (mode){
        case 0:
            #pragma omp parallel for num_threads(THREAD_NUMS) //default schedule
            for (int i = 0; i < M; ++i) {
                for (int k = 0; k< N; ++k) {
                    for (int j = 0; j < K; ++j) {
                        C[i * K + j] += A[i * N + k] * B[k * K + j];  //由于并行的是最外层循环，所以不需要对c数组的写进行枷锁处理，因为同一个地址只有一个线程会访问，线程内又是穿行的
                    }
                }
            } 
            break;
        case 1:
            #pragma omp parallel for num_threads(THREAD_NUMS) schedule(static, AVG_NUM) 
            for (int i = 0; i < M; ++i) {
                for (int k = 0; k< N; ++k) {
                    for (int j = 0; j < K; ++j) {
                        C[i * K + j] += A[i * N + k] * B[k * K + j];  //由于并行的是最外层循环，所以不需要对c数组的写进行枷锁处理，因为同一个地址只有一个线程会访问，线程内又是穿行的
                    }
                }
            } 
            break;
        case 2:
            #pragma omp parallel for num_threads(THREAD_NUMS) schedule(dynamic, 1) 
            for (int i = 0; i < M; ++i) {
                for (int k = 0; k< N; ++k) {
                    for (int j = 0; j < K; ++j) {
                        C[i * K + j] += A[i * N + k] * B[k * K + j];  //由于并行的是最外层循环，所以不需要对c数组的写进行枷锁处理，因为同一个地址只有一个线程会访问，线程内又是穿行的
                    }
                }
            } 
            break;                    
    }
    gettimeofday(&end, NULL);

    double time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    time_use /= 1000000;

    printf("Real time used: %.4f seconds\n", time_use);
    free(A);
    free(B);
    free(C);
    return 0;
}