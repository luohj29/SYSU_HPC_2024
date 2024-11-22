#include <pthread.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "matrix_multiply_s.h"
#include <sys/time.h>

int THREAD_NUM;
int SIZE;
int AVG_ROW;

float *A;
float *B;
float *C; // 全局指针变量

int M;
int N;
int K;

void *thread_matrix_multiply(void *arg) {
    int threadId = (intptr_t)arg;
    int rowStart = threadId * AVG_ROW;
    int rowEnd = (threadId == THREAD_NUM - 1) ? M : (threadId + 1) * AVG_ROW; // 最后一个线程处理剩余的行
    // printf("start: %d end: %d\n", rowStart, rowEnd);
    matrix_multiply_s(rowEnd - rowStart, N, K, A + N*rowStart, B, C +rowStart);
    pthread_exit(0); 
}

int main(int argc, char *argv[]) {
    struct timeval start;
    struct timeval end;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <matrix_size>" << std::endl;
        return 1;
    }

    THREAD_NUM = atoi(argv[1]);
    SIZE = atoi(argv[2]);
    AVG_ROW = SIZE / THREAD_NUM;
    M = N = K = SIZE;

    A = (float*)malloc(M * N * sizeof(float));
    B = (float*)malloc(N * K * sizeof(float));
    C = (float*)malloc(M * K * sizeof(float));

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    std::srand(static_cast<unsigned>(std::time(0))); // 设置随机数种子

    // 初始化随机矩阵A和B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)std::rand() / RAND_MAX * 10000.0;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i * K + j] = (float)std::rand() / RAND_MAX * 10000.0;
        }
    }
    gettimeofday(&start, NULL);
    pthread_t *pthread_set = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_create(&pthread_set[i], NULL, thread_matrix_multiply, (void *)(intptr_t)i);
    }

    float start_time = clock();
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(pthread_set[i], NULL);
    }
    gettimeofday(&end, NULL);
    float end_time = clock();

    printf("Cpu time used: %.4f seconds\n", (end_time - start_time) / CLOCKS_PER_SEC);


    double time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);// 微秒
    time_use /= 1000000;
    printf("Real time used: %.4f seconds\n", time_use);

    // 释放内存
    free(A);
    free(B);
    free(C);
    free(pthread_set);

    return 0;
}
