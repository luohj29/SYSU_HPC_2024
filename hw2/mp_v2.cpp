//本文件是MPi集合通信优化版
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include "matrix_multiply_s.h"

int main(int argc, char *argv[]) {
    int my_rank, comm_sz; // MPI进程的rank和进程数
    MPI_Init(NULL, NULL);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    // 从命令行获取矩阵的维度
    int M, N, K; // 矩阵的维度
    M = atoi(argv[1]);
    N = M;
    K = M;
    
    int split_num = M/(comm_sz);  //平均分的行数
    if (M % (comm_sz) !=0){
        printf("Error for comm_size!");
        return 0;
    }

    float *A = (float *)malloc(M * N * sizeof(float));
    float *localA = (float *)malloc(split_num * N * sizeof(float));
    float *B = (float *)malloc(N * K * sizeof(float));
    float *localC = (float *)malloc(split_num * K * sizeof(float));
    float *C = (float *)malloc(M * K * sizeof(float));
    double start ,end;
    if(my_rank==0){ //0号接受数据
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++){
                A[i*N +j] = (float)rand() / RAND_MAX * 10000.0;
            }
        }
        for (int i = 0; i < N; i++) {
            for(int j = 0; j < K; j++){
                B[i*K +j] = (float)rand() / RAND_MAX * 10000.0;
            }
        }
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < K; j++){
                C[i*K+j] = 0.0;
            }
        }
        start = MPI_Wtime();
    }

    if(comm_sz==1){ //串行
        matrix_multiply_s(M, K, N, A, B, C);
        end = MPI_Wtime();
        printf("computing time: %.8lf\n", end-start);
    } 
    else{ // 0号进程负责分配任务


        MPI_Bcast(B, N * K, MPI_FLOAT, 0,  MPI_COMM_WORLD); // 发送整个B矩阵
        MPI_Scatter(A, split_num*N, MPI_FLOAT, localA, split_num*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        matrix_multiply_s(split_num, N, K, localA, B, localC); // 计算

        MPI_Gather(localC, split_num * K, MPI_FLOAT, C, split_num * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
        if(my_rank ==0 ){
            double end = MPI_Wtime();
            printf("computing time: %.8lf\n", end-start);
            // printf("%f,%f,%f,%f", C[0], C[K-1], C[(M-1)*K],C[(M-1)*K+K-1]);
        }

        free(A);
        free(B);
        free(C);
        free(localC);
        free(localA);

    } 

    MPI_Finalize();
    return 0;
}