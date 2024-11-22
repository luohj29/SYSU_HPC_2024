/*
1）	基于pthreads的多线程库提供的基本函数，如线程创建、线程join、线程同步等，构建parallel_for函数，该函数实现对循环分解、分配和执行机制，
函数参数包括但不限于(int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads)；
其中start为循环开始索引；end为结束索引；increment每次循环增加索引数；functor为函数指针，指向被并行执行的循环代码块；arg为functor的入口参数；num_threads为并行线程数
*/

#include "parallel.h"
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
# include <math.h>
void *value_map_matrix(void *args)
{ // a function to map value to a matrix
    
    struct index_in *index_in = (struct index_in *)args;
    struct value2matrix *argues = (struct value2matrix *)index_in->args;
    int i, j;
    int row_increment = index_in->index_diy -> row_increment;
    int col_increment = index_in->index_diy->col_increment;
    int row_start = index_in->index_diy->row_start;
    int col_start = index_in->index_diy->col_start;
    int row_end = index_in->index_diy->row_end;
    int col_end = index_in->index_diy->col_end;
    // printf("%.4lf", argues->W[50][50]);
    // printf("%d, %d, %d\n", index_in->index_diy->col_start, index_in->index_diy->col_end, index_in->index_diy->col_increment);
    // printf("%p %.4lf\n", argues->W, argues->value);

    for (i = row_start; i < row_end; i += row_increment)
    {  
        for (j = col_start; j < col_end;j+=col_increment)
        {
            (*(argues->W))[i][j] = argues->value;
        }
    }
    return NULL;
}

void *matrix_map_matrix(void *args){ //a function to map value to a matrix
    struct index_in *index_in = (struct index_in *)args;
    struct matrix2matrix *argues = (struct matrix2matrix *)index_in->args;
    int i, j;
    int row_increment = index_in->index_diy -> row_increment;
    int col_increment = index_in->index_diy->col_increment;
    int row_start = index_in->index_diy->row_start;
    int col_start = index_in->index_diy->col_start;
    int row_end = index_in->index_diy->row_end;
    int col_end = index_in->index_diy->col_end;
    // printf("%.4lf", argues->W[50][50]);
    // printf("%d, %d, %d\n", index_in->index_diy->row_start, index_in->index_diy->row_end, index_in->index_diy->row_increment);
    // printf("%p %.4lf\n", argues->W, argues->value);

    for (i = row_start; i < row_end; i += row_increment)
    {  
        for (j = col_start; j < col_end;j+=col_increment)
        {
            (*(argues->dst))[i][j] = (*(argues->src))[i][j] ;
        }
    }
    return NULL;
}

void *matrix_funct_matrix(void *args)
{ // a function to map value to a matrix  
    struct index_in *index_in = (struct index_in *)args;
    struct matrix2matrix *argues = (struct matrix2matrix *)index_in->args;
    int i, j;
    int row_increment = index_in->index_diy -> row_increment;
    int col_increment = index_in->index_diy->col_increment;
    int row_start = index_in->index_diy->row_start;
    int col_start = index_in->index_diy->col_start;
    int row_end = index_in->index_diy->row_end;
    int col_end = index_in->index_diy->col_end;
    // printf("%.4lf", argues->W[50][50]);
    // printf("%d, %d, %d\n", index_in->index_diy->col_start, index_in->index_diy->col_end, index_in->index_diy->col_increment);
    // printf("%p %.4lf\n", argues->W, argues->value);

    for (i = row_start; i < row_end; i += row_increment)
    {  
        for (j = col_start; j < col_end;j+=col_increment)
        {
            (*(argues->dst))[i][j] = ((*(argues->src))[i-1][j] + (*(argues->src))[i+1][j] + (*(argues->src))[i][j-1] + (*(argues->src))[i][j+1])/4.0;
        }
    }
    return NULL;
}

void *find_matrixs_max_diff(void *args){
    struct index_in *index_in = (struct index_in *)args;
    struct matrix_matrix_diff *argues = (struct matrix_matrix_diff *)index_in->args;
    int i, j;
    int row_increment = index_in->index_diy -> row_increment;
    int col_increment = index_in->index_diy->col_increment;
    int row_start = index_in->index_diy->row_start;
    int col_start = index_in->index_diy->col_start;
    int row_end = index_in->index_diy->row_end;
    int col_end = index_in->index_diy->col_end;
    // printf("%.4lf", argues->W[50][50]);
    // printf("%d, %d, %d\n", index_in->index_diy->col_start, index_in->index_diy->col_end, index_in->index_diy->col_increment);
    // printf("%p %.4lf\n", argues->W, argues->value);

    for (i = row_start; i < row_end; i += row_increment)
    {  
        for (j = col_start; j < col_end;j+=col_increment)
        {
            if(fabs((*(argues->m1))[i][j]-(*(argues->m2))[i+1][j]) < *(argues->max_diff)){
                *(argues->max_diff) = fabs((*(argues->m1))[i][j] - (*(argues->m2))[i + 1][j]);
            }
        }
    }
    return NULL;  
}
int parallel_for(struct index *index_in, void *(*functor)(void*), void *arg , int num_threads){
    struct timeval start_time;
    struct timeval end_time;
    
    int row_start_i = index_in ->row_start;
    int row_end_i = index_in->row_start;
    int col_start_i = index_in->col_start;
    int col_end_i = index_in->col_end;
    int avg = (index_in->row_end - row_start_i) / num_threads;
    // printf("%p\n", functor);
    gettimeofday(&start_time, NULL);
    struct index_in index_in_set[num_threads];
    struct index index_set[num_threads];

    for (int i = 0; i < num_threads; i++) {
        row_end_i += avg;
        if (i == num_threads - 1)
        {
            row_end_i = index_in->row_end; //compute the left
        }
        index_set[i].row_start = row_start_i;
        index_set[i].row_end = row_end_i;
        index_set[i].col_start = col_start_i;
        index_set[i].col_end = col_end_i;
        index_set[i].row_increment = 1;
        index_set[i].col_increment = 1;
        index_in_set[i].index_diy = &index_set[i];
        index_in_set[i].args = arg;
        
        
        row_start_i = row_end_i;
    }

    gettimeofday(&end_time, NULL);
    for (int i = 0; i < num_threads; i++) {
        pthread_join(pthread_set[i], NULL);
    }
    double time_use=(end_time.tv_sec-start_time.tv_sec)*1000000+(end_time.tv_usec-start_time.tv_usec);
    time_use /= 1000000;
    // printf("Real time used: %.4f seconds\n", time_use);
    free(pthread_set);

    return 1;
}