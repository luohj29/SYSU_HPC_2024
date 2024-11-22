#ifndef PARARREL_H
#define PARARREL_H

struct for_index{
    int start;
    int end;
    int increment;
    float * A,*B,*C;
};

int parallel_for(float * A,float*B,float*C,int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads);
#endif