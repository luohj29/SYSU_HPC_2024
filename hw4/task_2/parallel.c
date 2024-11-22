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

int parallel_for(float * A,float*B,float*C,int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads){
    struct timeval start_time;
    struct timeval end_time;
    pthread_t *pthread_set = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    if(!pthread_set){
        printf("threads malloc failed!\n");
        return 0;
    }
    int avg = (end-start)/num_threads;
    int start_i = start;
    int end_i = start;

    gettimeofday(&start_time, NULL);
    struct for_index index_set[num_threads];
    // struct for_index *index_set = (struct for_index *)malloc(sizeof(struct for_index)* num_threads); //need malloc
    for (int i = 0; i < num_threads; i++) {
        end_i += avg;
        if(i==num_threads-1){
            end_i = end; //compute the left
        }
        index_set[i].start = start_i;
        index_set[i].end = end_i;
        index_set[i].increment = increment;
        index_set[i].A = A;
        index_set[i].B = B;
        index_set[i].C = C;
        pthread_create(&pthread_set[i], NULL, functor, (void*)&index_set[i]); //开启并行
        start_i = end_i;
    }

    gettimeofday(&end_time, NULL);
    for (int i = 0; i < num_threads; i++) {
        pthread_join(pthread_set[i], NULL);
    }
    double time_use=(end_time.tv_sec-start_time.tv_sec)*1000000+(end_time.tv_usec-start_time.tv_usec);
    time_use /= 1000000;
    // printf("Real time used: %.4f seconds\n", time_use);

    return 1;
}