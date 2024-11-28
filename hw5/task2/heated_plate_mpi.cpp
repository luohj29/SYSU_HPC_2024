# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <mpi.h>
#include <time.h>
#include <sys/time.h>
extern int parallel_for(struct index *index_in, void *(*functor)(void*), void *arg , int num_threads);
pthread_mutex_t shared_var;

int main ( int argc, char *argv[] );
int M, N;
void printf_matrix(double *matrix, int start, int end, int N) {
    // 确保输入参数合法性
    if (!matrix || start < 0 || end <= start || N <= 0) {
        printf("Invalid input parameters\n");
        return;
    }

    printf("Matrix elements from row %d to %d (total columns: %d):\n", start, end, N);

    // 遍历指定范围的行
    for (int i = start; i < end; i++) {
        // 遍历每行的列
        for (int j = 0; j < N; j++) {
            // 计算线性索引并打印矩阵元素
            printf("%8.2f ", matrix[i * N + j]);
        }
        // 打印完一行后换行
        printf("\n");
    }
}
/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_OPENMP.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programstd::ming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/


{
    int my_rank, comm_sz; // MPI进程的rank和进程数
    int signal;
    MPI_Init(NULL, NULL);
    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    if(argc!=3){
        printf("args num should be 2, M and N n");
        return 0;
    }
    M = atoi(argv[1]);
    N = atoi(argv[2]);

    struct timeval start_time;
    struct timeval end_time;

    int begin_Arow, end_Arow, avg_rows;
    if (comm_sz > 1) avg_rows = M / (comm_sz - 1);
if (my_rank ==0){
    double diff;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;

// 分配连续内存
    double *w = (double *)malloc(M * N * sizeof(double));

    // 检查内存分配是否成功
    if (!w ) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // 设置边界值
    for (int i = 1; i < M - 1; i++) {
        w[i * N + 0] = 100.0;         // 第一列
        w[i * N + (N - 1)] = 100.0;   // 最后一列
    }
    for (int j = 0; j < N; j++) {
        w[(M - 1) * N + j] = 100.0;   // 最后一行
        w[0 * N + j] = 0.0;           // 第一行
    }

    // 计算边界均值 mean
    for (int i = 1; i < M - 1; i++) {
        mean += w[i * N + 0] + w[i * N + (N - 1)];
    }
    for (int j = 0; j < N; j++) {
        mean += w[(M - 1) * N + j] + w[0 * N + j];
    }
    mean = mean / (double)(2 * M + 2 * N - 4);

    // 初始化矩阵内部值为 mean
    for (int i = 1; i < M - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            w[i * N + j] = mean;
        }
    }

    iterations = 0;
    iterations_print = 1;
    diff = epsilon;
    double my_diff[comm_sz - 1] = {0};
    gettimeofday(&start_time, NULL);
    while (epsilon <= diff)
    {
        diff = 0;
        // 发送矩阵u, w，让其他奴隶进程计算Mean, 有限元表格计算，赋值，计算max_diff
        for (int i = 0; i < comm_sz - 1; i++) { // 分配矩阵A的行
            // my_diff[i] = 0;
            begin_Arow = std::max(i * avg_rows - 1, 0);                                                                                                                                                                                                                                                                                                                           
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            end_Arow = std::min(end_Arow + 1, M);
            // 为了使用周围的数据，需要将区域上下两行多的数据传送过去。
            
            MPI_Send(&end_Arow, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD); // end_Arow可以作为信号，如果是-1，则奴隶进程结束
            // printf("0 processes send to process #%d form %d  to %d\n", i, begin_Arow, end_Arow);
            MPI_Send(&w[begin_Arow * N], (end_Arow - begin_Arow) * N, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
            // printf_matrix(w, begin_Arow, end_Arow, N);
            // for the u, it is the same as w at first
        }
        for (int i = 0; i < comm_sz - 1; i++) {  //return the std::max diff[i]
            begin_Arow = std::max(i * avg_rows - 1, 0);                                                                                                                                                                                                                                                                                                                           
            end_Arow = (i + 1 == comm_sz - 1) ? M : (i + 1) * avg_rows;
            end_Arow = std::min(end_Arow + 1, M);
            MPI_Recv(&my_diff[i], 1, MPI_DOUBLE, i + 1, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&w[(begin_Arow+1) * N], (end_Arow - begin_Arow -2) * N, MPI_DOUBLE, i + 1, 4, MPI_COMM_WORLD, &status);
            diff = std::max(my_diff[i], diff);
        }
        // printf("0 process received w\n");
        // printf_matrix(w, 0, M, N);
        // while(1)

        iterations++;
        if ( iterations == iterations_print )
        {
            printf ( "  %8d  %f\n", iterations, diff );
            iterations_print = 2 * iterations_print;
        }
    }
    signal = -1;
    printf("At %d , episilon err reached, gonna kill all processes\n", iterations);
    for(int i = 0; i < comm_sz - 1; i++){
        MPI_Send(&signal, 1, MPI_INT, i + 1, 10, MPI_COMM_WORLD); // end_Arow可以作为信号，如果是-1，则奴隶进程结束
    }
    gettimeofday(&end_time, NULL);
    long seconds, useconds;
    double elapsed_time;
    seconds = end_time.tv_sec - start_time.tv_sec;
    useconds = end_time.tv_usec - start_time.tv_usec;
    elapsed_time = seconds + useconds / 1000000.0;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
    free(w);
}

else { // 非 0 号进程的代码 所有的索引满足左闭右开要求
    int real_begin_Arow = my_rank == 1 ? 1 : avg_rows * (my_rank - 1);
    int virtual_begin_Arow = real_begin_Arow - 1;
    int real_end_Arow = my_rank == comm_sz - 1 ? M - 1 : (avg_rows * my_rank);
    int virtual_end_Arow = real_end_Arow + 1;
    double *local_w = (double *)malloc((virtual_end_Arow - virtual_begin_Arow) * N * sizeof(double)); // 分配子矩阵 w 的内存
    double *local_u = (double *)malloc((virtual_end_Arow - virtual_begin_Arow) * N * sizeof(double)); // 分配子矩阵 u 的内存
    double max_diff;
    int M_len = virtual_end_Arow - virtual_begin_Arow;
    while (true)
    {
        // 接收分配的行信息，判断是否结束
        MPI_Recv(&signal, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        // printf("rank: %d recieved signal: %d and virtual_end_arow is %d\n", my_rank, signal, virtual_end_Arow);
        if (virtual_end_Arow != signal)
        { // 结束信号
            printf("%d process received quit signal %d\n", my_rank, signal);
            break;
        }

        // 接收子矩阵数据
        MPI_Recv(local_u, (virtual_end_Arow - virtual_begin_Arow) * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        int actual_count;
        MPI_Get_count(&status, MPI_DOUBLE, &actual_count);
        // printf("Expected                                : %d, Received: %d\n", (virtual_end_Arow - virtual_begin_Arow) * N, actual_count);
        // printf("%d\n", my_rank);
        // printf_matrix(local_u, 0, M_len, N);

        max_diff = 0.0;
        for (int i = 0; i < M_len; i++)
        {
            if(i == 0 || i ==M_len-1){
                for (int j = 0; j < N ; j++) {
                    local_w[i * N + j] = local_u[i * N + j];
                }
                continue;
            }
            for (int j = 0; j < N ; j++) {
                if(j == 0||j==N-1){
                    local_w[i * N + j] = local_u[i * N + j];
                    continue;
                }
                local_w[i * N + j] = 0.25 * (local_u[(i - 1) * N + j] + local_u[(i + 1) * N + j] + 
                                             local_u[i * N + (j - 1)] + local_u[i * N + (j + 1)]);
                double diff = fabs(local_u[i * N + j] - local_w[i * N + j]);
                if (diff > max_diff) {
                    max_diff = diff; // 更新最大差异
                }
            }
        }
        // printf("rank: %d max_diff: %.4lf\n", my_rank, max_diff);
        // printf_matrix(local_w, 0, M_len, N);
        // while(1){}
        // 返回结果

        // while(1)
        MPI_Send(&max_diff, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);                                                         // 发送最大差异
        MPI_Send(&local_w[N], (real_end_Arow - real_begin_Arow) * N, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD); // 发送更新后的子矩阵 w
    }
    printf("%d process ending\n", my_rank);
    free(local_w); // 释放子矩阵内存
    free(local_u);
}

  MPI_Finalize();
  return 0;
      // 释放内存
}
