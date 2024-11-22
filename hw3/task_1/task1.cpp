#include <pthread.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#define THREAD_NUM 8
/*
  优化思路: 各个线程读取要处理的下标(连续),然后走出临界区,读取数据,求和,放到自己的local_sum堆区(不能直接求和,会有数据冒险),等到所有线程回收,统一将local_sum全部加起来 
*/

const int ARRAY_SIZE = 100000;
const int MAX_ELEMENTS = 10;
float *Numbers; //数组
int Global_index = 0; // 共享变量
float Sum = 0;        // 共享结果
float Local_sum[THREAD_NUM] = {0};
pthread_mutex_t mutex;



// 连续读取10个数字
void *thread_Numbers_Add(void *arg) {
    while (Global_index <= ARRAY_SIZE) {
        int rank = (intptr_t)arg;
        pthread_mutex_lock(&mutex); // 进入临界区
        if (Global_index >= ARRAY_SIZE) {
            pthread_mutex_unlock(&mutex);
            break; // 如果已处理所有元素，退出循环
        }
        int start = Global_index;
        int end = start + std::min(MAX_ELEMENTS, ARRAY_SIZE - Global_index); // 确保不越界
        Global_index += end - start;
        pthread_mutex_unlock(&mutex); // 退出临界区

        for (int i = start; i < end; i++) {
            Local_sum[rank] += Numbers[i]; // 求和操作
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("hello");
    double totle_time = 0;
    Numbers = (float *)malloc(ARRAY_SIZE * sizeof(float)); // 开辟1000个数组空间

    // 初始化数组
    for (int i = 0; i < ARRAY_SIZE; i++) {
        Numbers[i] = i + 1.0 ;
    }

    int Test_Rnd = 10; //计算10次取平均
    int ori = Test_Rnd;
    struct timeval start, end;

    while(Test_Rnd --){
        pthread_mutex_init(&mutex, NULL); // 初始化互斥锁
        pthread_t *pthread_set = (pthread_t *)malloc(THREAD_NUM * sizeof(pthread_t)); // 开辟多线程

        for (int i = 0; i < THREAD_NUM; i++) {
            pthread_create(&pthread_set[i], NULL, thread_Numbers_Add, (void *)(intptr_t)i);
        }
        gettimeofday(&start, NULL);

        // 等待所有线程完成
        for (int i = 0; i < THREAD_NUM; i++) {
            pthread_join(pthread_set[i], NULL);
            Sum += Local_sum[i];
            Local_sum[i] = 0;
        }

        gettimeofday(&end, NULL);

        double time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec); // 微秒
        time_use /= 1000000;
        printf("Index: %d, Sum: %.2f, Time: %.8f\n", ori - Test_Rnd, Sum, time_use);
        totle_time += time_use;
        free(pthread_set);
        Sum = 0;
        Global_index = 0; //恢复数据为0
        }

    double avg_time = totle_time/ori;
    printf("Continuous index of %d use avg time: %.8f \n", MAX_ELEMENTS, avg_time);
    free(Numbers);
    pthread_mutex_destroy(&mutex); // 销毁互斥锁
    return 0; 
}
