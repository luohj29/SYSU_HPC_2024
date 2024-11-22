/*
1）	基于pthreads的多线程库提供的基本函数，如线程创建、线程join、线程同步等，构建parallel_for函数，该函数实现对循环分解、分配和执行机制，
函数参数包括但不限于(int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads)；
其中start为循环开始索引；end为结束索引；increment每次循环增加索引数；functor为函数指针，指向被并行执行的循环代码块；arg为functor的入口参数；num_threads为并行线程数
*/

#include "parallel.h"

PthreadPool::PthreadPool()
{
    thread_num = 0;
    running_num = 0;
    shutdown = 0;
}

PthreadPool::~PthreadPool()
{
    pthread_mutex_lock(&lock);    // 先上锁， 防止有程序占用
    pthread_mutex_destroy(&lock); // 销毁
    pthread_cond_destroy(&notify);
    delete[] threads;
}

int PthreadPool::Init(unsigned int num)
{
    // 初始化互斥锁和条件变量
    do
    {
        if (num <= 0)
            break;
        if (pthread_mutex_init(&lock, NULL))
            break;
        if (pthread_cond_init(&notify, NULL))
            break;

        // 初始化线程数组
        threads = new pthread_t[num];

        // 创建线程
        for (int i = 0; i < num; i++)
        {
            if (pthread_create(threads + i, NULL, threadpool_thread, (void *)this) != 0)
            {
                // 创建不成功则销毁
                Destory();
                break;
            }
            running_num++;
            thread_num++;
        }
        return 0; // 成功

    } while (0);
    thread_num = 0;
    return Pthreadpool_invalid;
}

int PthreadPool::Destory(PthreadPool_Shutdown flag)
{
    do
    {
        // 取得互斥锁资源
        if (pthread_mutex_lock(&lock) != 0)
        {
            return Pthreadpool_lock_failure;
        }

        shutdown = flag; // 标记标记

        /* 唤醒所有因条件变量阻塞的线程，并释放互斥锁 */
        if ((pthread_cond_broadcast(&notify) != 0) || (pthread_mutex_unlock(&lock) != 0))
            break;

        /* 等待所有线程结束 */
        for (int i = 0; i < thread_num; i++)
            if (pthread_join(threads[i], NULL) != 0)
                break;
        return 0;
    } while (0);
    return -1;
}

int PthreadPool::AddTask(void* (*function)(void *), void *argument)
{
    printf("In AddTask: %p\n", function);    
    if (thread_num == 0 || function == NULL)
        return Pthreadpool_invalid;
    /* 必须先取得互斥锁所有权 */
    if (pthread_mutex_lock(&lock) != 0)
        return Pthreadpool_lock_failure;

    // 检查是否关闭了线程池
    if (shutdown)
    {
        return Pthreadpool_shutdown;
    }

    // 新加入
    Pthreadpool_Runable newRunable;
    newRunable.function = function;
    newRunable.argument = argument;
    // 加入队列
    thread_queue.push(newRunable);
    
    // 发出signal
    if (pthread_cond_signal(&notify) != 0)
        return Pthreadpool_lock_failure;
    pthread_mutex_unlock(&lock);
    return 0;
}

void PthreadPool::pool_wait() {
    pthread_mutex_lock(&this->lock);
    while (!this->shutdown || this->waiting_num > 0) { // 等待所有任务完成
        printf("waiting for threads\n");
        pthread_cond_wait(&this->notify, &this->lock);
    }
    pthread_mutex_unlock(&this->lock);
}


// 线程运行函数
void *PthreadPool::threadpool_thread(void *threadpool)
{
    // printf("hello\n");
    PthreadPool *pool = (PthreadPool *)threadpool; // 获取当前实例
    while (1)
    {
        /* 取得互斥锁资源 */
        pthread_mutex_lock(&(pool->lock));
        while ((pool->thread_queue.empty()) && (!pool->shutdown))
        {
            /* 任务队列为空，且线程池没有关闭时阻塞在这里 */
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        /* 关闭的处理 */
        if((pool->shutdown == immediate_shutdown) ||
           ((pool->shutdown == graceful_shutdown) && (pool->thread_queue.empty()))) {
            break;
        }

        // 取队列中的任务
        Pthreadpool_Runable Runable;
        if (!pool->thread_queue.empty())
        {
            Runable = pool->thread_queue.front();
            pool->thread_queue.pop(); // 出队
        }

        // running_num
        pool->waiting_num--;

        /* 释放互斥锁 */
        pthread_mutex_unlock(&(pool->lock));

        // 开始运行任务
        printf("In working threads: %p\n", Runable.function);
        (*(Runable.function))(Runable.argument);
        // 结束，回到等待
    }
    // 更新正在运行的线程数
    pool->running_num--;

    /* 释放互斥锁 */
    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);
    return (NULL);
}

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
    printf("In parallel for: %p\n", functor);
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

        threadspool.AddTask(functor, (void*) &index_in_set[i]);  //加入到工作任务队列

        row_start_i = row_end_i;
    }
    threadspool.pool_wait(); // wait all the batch to finish!

    gettimeofday(&end_time, NULL);

    double time_use=(end_time.tv_sec-start_time.tv_sec)*1000000+(end_time.tv_usec-start_time.tv_usec);
    time_use /= 1000000;
    // printf("Real time used: %.4f seconds\n", time_use);

    return 1;
}