#include <iostream>
#include <queue>
#include <algorithm>
#include <unistd.h>

using namespace std;

// ����ṹ��
struct Pthreadpool_Runable
{
    void* (*function)(void *);
    void *argument;
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
    pthread_cond_t notify;                            // ��������
    queue<Pthreadpool_Runable> thread_queue;          // �������
    pthread_t *threads;                               // ��������
    int shutdown;                                     // ��ʾ�̳߳��Ƿ�ر�
    static void *threadpool_thread(void *threadpool); // ���к���

public:
    PthreadPool();
    ~PthreadPool();
    int thread_num;                                                  // �߳�����
    int running_num;                                                 // �������е��߳���
    int waiting_num;                                                 // �����еȴ�����Ŀ
    int Init(unsigned int num);                                      // ��ʼ���̳߳�
    int AddTask(void* (*function)(void *), void *argument = nullptr); // ��������
    int Destory(PthreadPool_Shutdown flag = graceful_shutdown);      // ֹͣ���ڽ��е����񲢴ݻ��̳߳�
};