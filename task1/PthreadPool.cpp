#include "PthreadPool.h"

PthreadPool::PthreadPool()
{
    thread_num = 0;
    running_num = 0;
    shutdown = 0;
}

PthreadPool::~PthreadPool()
{
    pthread_mutex_lock(&lock);    // �������� ��ֹ�г���ռ��
    pthread_mutex_destroy(&lock); // ����
    pthread_cond_destroy(&notify);
    delete[] threads;
}

int PthreadPool::Init(unsigned int num)
{
    // ��ʼ������������������
    do
    {
        if (num <= 0)
            break;
        if (pthread_mutex_init(&lock, NULL))
            break;
        if (pthread_cond_init(&notify, NULL))
            break;

        // ��ʼ���߳�����
        threads = new pthread_t[num];

        // �����߳�
        for (int i = 0; i < num; i++)
        {
            if (pthread_create(threads + i, NULL, threadpool_thread, (void *)this) != 0)
            {
                // �������ɹ�������
                Destory();
                break;
            }
            running_num++;
            thread_num++;
        }
        return 0; // �ɹ�

    } while (0);
    thread_num = 0;
    return Pthreadpool_invalid;
}

int PthreadPool::Destory(PthreadPool_Shutdown flag)
{
    do
    {
        // ȡ�û�������Դ
        if (pthread_mutex_lock(&lock) != 0)
        {
            return Pthreadpool_lock_failure;
        }

        shutdown = flag; // ��Ǳ��

        /* �������������������������̣߳����ͷŻ����� */
        if ((pthread_cond_broadcast(&notify) != 0) || (pthread_mutex_unlock(&lock) != 0))
            break;

        /* �ȴ������߳̽��� */
        for (int i = 0; i < thread_num; i++)
            if (pthread_join(threads[i], NULL) != 0)
                break;
        return 0;
    } while (0);
    return -1;
}

int PthreadPool::AddTask(void* (*function)(void *), void *argument)
{

    if (thread_num == 0 || function == NULL)
        return Pthreadpool_invalid;
    /* ������ȡ�û���������Ȩ */
    if (pthread_mutex_lock(&lock) != 0)
        return Pthreadpool_lock_failure;

    // ����Ƿ�ر����̳߳�
    if (shutdown)
    {
        return Pthreadpool_shutdown;
    }

    // �¼���
    Pthreadpool_Runable newRunable;
    newRunable.function = function;
    newRunable.argument = argument;
    // �������
    thread_queue.push(newRunable);

    // ����signal
    if (pthread_cond_signal(&notify) != 0)
        return Pthreadpool_lock_failure;
    pthread_mutex_unlock(&lock);
    return 0;
}

// �߳����к���
void *PthreadPool::threadpool_thread(void *threadpool)
{
    PthreadPool *pool = (PthreadPool *)threadpool; // ��ȡ��ǰʵ��
    while (1)
    {
        /* ȡ�û�������Դ */
        pthread_mutex_lock(&(pool->lock));
        while ((pool->thread_queue.empty()) && (!pool->shutdown))
        {
            /* �������Ϊ�գ����̳߳�û�йر�ʱ���������� */
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        /* �رյĴ��� */
        if((pool->shutdown == immediate_shutdown) ||
           ((pool->shutdown == graceful_shutdown) && (pool->thread_queue.empty()))) {
            break;
        }

        // ȡ�����е�����
        Pthreadpool_Runable Runable;
        if (!pool->thread_queue.empty())
        {
            Runable = pool->thread_queue.front();
            pool->thread_queue.pop(); // ����
        }

        // running_num
        pool->waiting_num--;

        /* �ͷŻ����� */
        pthread_mutex_unlock(&(pool->lock));

        // ��ʼ��������
        (*(Runable.function))(Runable.argument);
        // �������ص��ȴ�
    }
    // �����������е��߳���
    pool->running_num--;

    /* �ͷŻ����� */
    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);
    return (NULL);
}