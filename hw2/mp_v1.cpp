#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>
#include "matrix_multiply_s.h"
#include <ctime>

void printmatrix(int M, int N, float *A){
    for(int i=0;i<M;i++){
        for(int j=0; j<N;j++){
            printf("%f ", A[i*N+j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int my_rank, comm_sz; // MPI进程的rank和进程数
    MPI_Init(NULL, NULL);
    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    // 从命令行获取矩阵的维度
    int M, N, K; // 矩阵的维度
    M = atoi(argv[1]);
    N = M;
    K = M;
    // printf("start of mpi %d\n", my_rank);
    int begin_Arow, end_Arow, avg_rows;
    if (comm_sz > 1) avg_rows = M / (comm_sz - 1);

    if (my_rank == 0) { // 0号进程负责分配任务

        // fflush(stdout);

        // 创建随机矩阵

        float *A = (float *)malloc(M * N * sizeof(float));
        float *B = (float *)malloc(N * K * sizeof(float));
        float *C = (float *)malloc(M * K * sizeof(float));


        for (int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++){
                A[i*N +j] = (float) std::rand() / RAND_MAX * 10000.0;//
                //  A[i*N +j] = 1;
            }
        }
        for (int i = 0; i < N; i++) {
            for(int j = 0; j < K; j++){
                B[i*K +j] = (float) std::rand() / RAND_MAX * 10000.0;
                // B[i*K +j] = 1;
            }
        }
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < K; j++){
                C[i*K+j] = 0.0;
            }
        }

        // 初始化MPI,计算执行时间
        double start_time = MPI_Wtime();
        if (comm_sz==1){ //serial
            matrix_multiply_s(M,N,K,A,B,C);
            double end_time= MPI_Wtime();
            // printmatrix(M,K,C);
            printf("computing time: %.8lf\n", end_time-start_time);
            MPI_Finalize();
            return 0;
        }
        for (int i = 0; i < comm_sz - 1; i++) { // 分配矩阵A的行
            begin_Arow = i * avg_rows;
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            MPI_Send(&end_Arow, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD);
            MPI_Send(&A[begin_Arow * N], (end_Arow - begin_Arow) * N, MPI_FLOAT, i + 1, 1, MPI_COMM_WORLD);
            MPI_Send(B, N * K, MPI_FLOAT, i + 1, 2, MPI_COMM_WORLD); // 发送整个B矩阵
        }

        for (int i = 0; i < comm_sz - 1; i++) {
            begin_Arow = avg_rows * i;
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            MPI_Recv(&C[begin_Arow * K], (end_Arow - begin_Arow) * K, MPI_FLOAT, i + 1, 3, MPI_COMM_WORLD, &status);
        }
        double end_time = MPI_Wtime();
        // // 输出结果矩阵C
        printf("computing time: %.8lf\n", end_time-start_time);

        // printmatrix(M, K, C);

        free(A);
        free(B);
        free(C);
    } else { // 其他进程接收数据并计算
        MPI_Recv(&end_Arow, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        begin_Arow = avg_rows * (my_rank - 1);

        float *localA = (float *)malloc((end_Arow - begin_Arow) * N * sizeof(float));
        float *localB = (float *)malloc(N * K * sizeof(float));
        float *localC = (float *)malloc((end_Arow - begin_Arow) * K * sizeof(float));

        MPI_Recv(localA, (end_Arow - begin_Arow) * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(localB, N * K, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);

        matrix_multiply_s(end_Arow - begin_Arow, N, K, localA, localB, localC); // 计算

        MPI_Send(localC, (end_Arow - begin_Arow) * K, MPI_FLOAT, 0, 3, MPI_COMM_WORLD); // 发送结果

        free(localA);
        free(localB);
        free(localC);
    }

    MPI_Finalize();


    // printf("end of mpi %d\n", my_rank);

    return 0;
}
