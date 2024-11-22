#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

int THREAD_NUMS, SIZE;
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("NEED 3 PARA!!!\n");
        return 1;
    }
    THREAD_NUMS = atoi(argv[1]);
    SIZE = atoi(argv[2]); 
    // int mode =  atoi(argv[3]);

    srand((unsigned)time(NULL));	//ʹ�õ�ǰϵͳʱ����Ϊ����
    // AVG_NUM = SIZE/THREAD_NUMS;
    int M=SIZE;
    int N=SIZE;
    int K=SIZE;;
    float A[M*N];
    float B[N*K];
    float C[M*K];
    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (float)rand() / RAND_MAX * 10000.0;;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            B[i * K + j] = (float)rand() / RAND_MAX * 10000.0;;
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
    #pragma omp parallel for num_threads(THREAD_NUMS) shared(A,B,C)//default schedule
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k< N; ++k) {
            for (int j = 0; j < K; ++j) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];  //���ڲ��е��������ѭ�������Բ���Ҫ��c�����д���м���������Ϊͬһ����ַֻ��һ���̻߳���ʣ��߳������Ǵ��е�
            }
        }
    } 
    gettimeofday(&end, NULL);

    double time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    time_use /= 1000000;

    printf("Real time used: %.4f seconds\n", time_use);
    return 0;
}