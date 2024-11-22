#include<stdlib.h>
# include "PthreadPool.h"
#ifndef PARARREL_H
#define PARARREL_H
# define M 500
# define N 500
/* struct graph
                    index_in
            /                   \
  index(for iterring)       void* args(for different args need of different functions)
                                value2matrix for mapping a value 2 a matrix
                                matrix2matrix for mapping a matrix 2 another matrix
*/

struct index{ //used to iter a matrix
    int row_start;
    int row_end;
    int row_increment;
    int col_start;
    int col_end;
    int col_increment;
};
struct index_in // used to iter a matrix and give a value!
{ 
    struct index *index_diy;
    void *args;
};

struct value2matrix{ 
    double value;
    double (*W)[][N];  //声明一个指向一定大小的二维数组的指针！
    // double *W;
};

struct matrix2matrix// used to map a matrix 2 a matrix
{ 
    double (*src)[][N];
    double (*dst)[][N];
};

struct matrix_matrix_diff
{
    double (*m1)[][N];
    double (*m2)[][N];
    double *max_diff;
};

void *value_map_matrix(void *args);

void *matrix_map_matrix(void *args);

void *matrix_funct_matrix(void *args);

void *find_matrixs_max_diff(void *args);
/*
replace the openmp parallel for
 */
int parallel_for(struct index *index_in, void *(*functor)(void*), void *arg , int num_threads);

#endif