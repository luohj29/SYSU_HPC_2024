# include "parallel.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
extern int parallel_for(struct index *index_in, void *(*functor)(void*), void *arg , int num_threads);
pthread_mutex_t shared_var;

int main ( int argc, char *argv[] );
int M, N;

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
    Parallel Programming in C with MPI and OpenMP,
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
  if(argc!=4){
      printf("args num should be 4\n");
      return 0;
  }
  M = atoi(argv[1]);
  N = atoi(argv[2]);
int threads_num = atoi(argv[3]);
double diff;
double epsilon = 0.001;
int i;
int iterations;
int iterations_print;
int j;
double mean;
// 动态分配二维数组 u 和 w
double **u = new double *[M]; // 为行分配内存
double **w = new double *[M]; // 为行分配内存

for (int i = 0; i < M; ++i)
{
    u[i] = new double[N]; // 为每一行分配列内存
    w[i] = new double[N]; // 为每一行分配列内存
    }
    double wtime;
    PthreadPool threadspool;
    threadspool.Init(threads_num);
 

//   printf ( "\n" );
//   printf ( "HEATED_PLATE_OPENMP\n" );
//   printf ( "  C/OpenMP version\n" );
//   printf ( "  A program to solve for the steady state temperature distribution\n" );
//   printf ( "  over a rectangular plate.\n" );
//   printf ( "\n" );
//   printf ( "  Spatial grid of %d by %d points.\n", M, N );
//   printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
//   printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
//   printf ( "  Number of threads =              %d\n", threads_num );
/*
  Set the boundary values, which don't change. 
*/
  mean = 0.0;

#pragma omp parallel shared ( w ) private ( i, j )
  {
#pragma omp for //to replace?
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][0] = 100.0;
    }
#pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][N-1] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[M-1][j] = 100.0;
    }
#pragma omp for
    for ( j = 0; j < N; j++ )
    {
      w[0][j] = 0.0;
    }
/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
#pragma omp for reduction ( + : mean )
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + w[i][0] + w[i][N-1];
    }
#pragma omp for reduction ( + : mean )
    for ( j = 0; j < N; j++ )
    {
      mean = mean + w[M-1][j] + w[0][j];
    }
  }
/*
  OpenMP note:
  You cannot normalize MEAN inside the parallel region.  It
  only gets its correct value once you leave the parallel region.
  So we interrupt the parallel region, set MEAN, and go back in.
*/
  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
//   printf ( "\n" );
//   printf ( "  MEAN = %f\n", mean );
/* 
  Initialize the interior solution to the mean value.
*/
  int num_threads = 15;
  struct index index_diy = {1, M - 1, 1, 1, N - 1, 1};
  struct value2matrix value2matrix_diy = {mean,w};
    // printf("%p\n", w);
    // printf("%.4lf\n", w[50][50]);
  parallel_for(&threadspool, &index_diy, value_map_matrix, (void *)&value2matrix_diy, num_threads);
//   iterate until the  new solution W differs from the old solution U
//   by no more than EPSILON.

  iterations = 0;
  iterations_print = 1;
//   printf ( "\n" );
//   printf ( " Iteration  Change\n" );
//   printf ( "\n" );
  wtime = omp_get_wtime ( );

  diff = epsilon;

  while ( epsilon <= diff )
  {
    struct index index_diy2 = {0, M , 1, 0, N , 1};
    struct matrix2matrix matrix2matrix_diy = {w, u};

    parallel_for(&threadspool, &index_diy2, matrix_map_matrix, (void *)&matrix2matrix_diy, num_threads);

/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/
    matrix2matrix_diy.dst = w, matrix2matrix_diy.src = u;
    struct index index_diy3 = {1, M-1 , 1, 1, N-1 , 1};
    parallel_for(&threadspool, &index_diy3, matrix_funct_matrix, (void *)&matrix2matrix_diy, num_threads);
    // printf("%.4lf ,%4lf\n", u[5][5], w[498][498]);
    // printf("%p %p\n", w, u);
    //   C and C++ cannot compute a maximum as a reduction operation.

    //   Therefore, we define a private variable MY_DIFF for each thread.
    //   Once they have all computed their values, we use a CRITICAL section
    //   to update DIFF.

    //diff represents the biggest differnce between u and w
    //using diy parallel for to find the max difference, inner function use mutex lock for critical write
    diff = 0.0;
    struct matrix_matrix_diff diff_diy = {w, u, &diff};
    // printf("%.4lf ,%4lf\n", u[5][5], w[498][498]);
    pthread_mutex_init(&shared_var, NULL);
    parallel_for(&threadspool, &index_diy3, find_matrixs_max_diff, (void *)&diff_diy, num_threads);
    // printf("%p %p\n", w, u);
    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }
  } 
  wtime = omp_get_wtime ( ) - wtime;

//   printf ( "\n" );
//   printf ( "  %8d  %f\n", iterations, diff );
//   printf ( "\n" );
//   printf ( "  Error tolerance achieved.\n" );
  printf ( "Wallclock time = %f\n", wtime );
/*
  Terminate.
*/
//   printf ( "\n" );
//   printf ( "HEATED_PLATE_OPENMP:\n" );
//   printf ( "  Normal end of execution.\n" );
//   for (int i = M-5;i<M;i++)
//   {
//       for (int j = N-5;j<N;j++){
//           printf("%.4lf ", w[i][j]);
//       }
//       printf("\n");
//   }
  threadspool.Destory();
    for (int i = 0; i < M; ++i) {
        delete[] u[i];  // 释放每一行的内存
        delete[] w[i];  // 释放每一行的内存
    }
  delete[] u;
  delete[] w;
  return 0;
      // 释放内存
#undef M
# undef N
}
