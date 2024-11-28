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

// 关闭方式定义
enum PthreadPool_Shutdown {
    graceful_shutdown  = 1,   // 等待线程结束后关闭
    immediate_shutdown = 2  // 立即关闭
};

// 错误码定义
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
    pthread_mutex_t lock;                             // 互斥锁
    pthread_cond_t notify;                            // 条件变量，标记是否有任务
    std::queue<Pthreadpool_Runable> thread_queue;          // 任务队列
    pthread_t *threads;                               // 任务数组
    int shutdown;                                     // 表示线程池是否关闭
    static void *threadpool_thread(void *threadpool); // 运行函数

public:
    PthreadPool();
    ~PthreadPool();
    bool is_queue_empty();
    int thread_num;                                                  // 线程数量
    int running_num;                                                 // 正在运行的线程数
    int waiting_num;                                                 // 队列中等待的数目
    int unfinished_task;
    pthread_cond_t cond;                                             // 标记一批任务是否完成
    int Init(unsigned int num);                                       // 初始化线程池
    int AddTask(void* (*function)(void *), void *argument = nullptr); // 加入任务
    int Destory(PthreadPool_Shutdown flag = graceful_shutdown);      // 停止正在进行的任务并摧毁线程池
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
    double **w;  //声明一个指向一定大小的二维数组的指针！
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