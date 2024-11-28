#ifndef PARARREL_H
#define PARARREL_H
#include <cstdlib>
#include <iostream>
#include <queue>
#include <algorithm>
#include <unistd.h>
#include <cstdio>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
# include <cmath>
#include <semaphore.h>
// # define M 2097152
// # define N 2097152


extern int M;
extern int N;
extern pthread_mutex_t shared_var;
struct Pthreadpool_Runable
{
    void* (*function)(void *);
    void* argument;
};

// �رշ�ʽ����
enum PthreadPool_Shutdown {
    graceful_shutdown  = 1,   // �ȴ��߳̽�����ر�
    immediate_shutdown = 2  // �����ر�
};

// �����붨��
enum PthreadPool_Error {
    Pthreadpool_invalid        = -1,
    Pthreadpool_lock_failure   = -2,
    Pthreadpool_queue_full     = -3,
    Pthreadpool_shutdown       = -4,
    Pthreadpool_thread_failure = -5
};

class PthreadPool
{
private:
    pthread_mutex_t lock;                             // ������
    pthread_cond_t notify;                            // ��������������Ƿ�������
    std::queue<Pthreadpool_Runable> thread_queue;          // �������
    pthread_t *threads;                               // ��������
    int shutdown;                                     // ��ʾ�̳߳��Ƿ�ر�
    static void *threadpool_thread(void *threadpool); // ���к���

public:
    PthreadPool();
    ~PthreadPool();
    bool is_queue_empty();
    int thread_num;                                                  // �߳�����
    int running_num;                                                 // �������е��߳���
    int waiting_num;                                                 // �����еȴ�����Ŀ
    int unfinished_task;
    pthread_cond_t cond;                                             // ���һ�������Ƿ����
    int Init(unsigned int num);                                       // ��ʼ���̳߳�
    int AddTask(void* (*function)(void *), void *argument = nullptr); // ��������
    int Destory(PthreadPool_Shutdown flag = graceful_shutdown);      // ֹͣ���ڽ��е����񲢴ݻ��̳߳�
    void pool_wait();  //wait every threads
};

extern PthreadPool threadspool;

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
    double **w;  //����һ��ָ��һ����С�Ķ�ά�����ָ�룡
    // double *W;
};

struct matrix2matrix// used to map a matrix 2 a matrix
{ 
    double **src;
    double **dst;
};

struct matrix_matrix_diff
{
    double **m1;
    double **m2;
    double *max_diff;
};

void *value_map_matrix(void *args);

void *matrix_map_matrix(void *args);

void *matrix_funct_matrix(void *args);

void *find_matrixs_max_diff(void *args);


/*
replace the openmp parallel for
 */
int parallel_for(PthreadPool *threadspool,  index *index_in, void *(*functor)(void*), void *arg , int num_threads);

#endif