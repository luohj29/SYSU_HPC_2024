#include <pthread.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#define THREAD_NUM 8
/*
  �Ż�˼·: �����̶߳�ȡҪ������±�(����),Ȼ���߳��ٽ���,��ȡ����,���,�ŵ��Լ���local_sum����(����ֱ�����,��������ð��),�ȵ������̻߳���,ͳһ��local_sumȫ�������� 
*/

const int ARRAY_SIZE = 100000;
const int MAX_ELEMENTS = 10;
float *Numbers; //����
int Global_index = 0; // �������
float Sum = 0;        // ������
float Local_sum[THREAD_NUM] = {0};
pthread_mutex_t mutex;



// ������ȡ10������
void *thread_Numbers_Add(void *arg) {
    while (Global_index <= ARRAY_SIZE) {
        int rank = (intptr_t)arg;
        pthread_mutex_lock(&mutex); // �����ٽ���
        if (Global_index >= ARRAY_SIZE) {
            pthread_mutex_unlock(&mutex);
            break; // ����Ѵ�������Ԫ�أ��˳�ѭ��
        }
        int start = Global_index;
        int end = start + std::min(MAX_ELEMENTS, ARRAY_SIZE - Global_index); // ȷ����Խ��
        Global_index += end - start;
        pthread_mutex_unlock(&mutex); // �˳��ٽ���

        for (int i = start; i < end; i++) {
            Local_sum[rank] += Numbers[i]; // ��Ͳ���
        }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("hello");
    double totle_time = 0;
    Numbers = (float *)malloc(ARRAY_SIZE * sizeof(float)); // ����1000������ռ�

    // ��ʼ������
    for (int i = 0; i < ARRAY_SIZE; i++) {
        Numbers[i] = i + 1.0 ;
    }

    int Test_Rnd = 10; //����10��ȡƽ��
    int ori = Test_Rnd;
    struct timeval start, end;

    while(Test_Rnd --){
        pthread_mutex_init(&mutex, NULL); // ��ʼ��������
        pthread_t *pthread_set = (pthread_t *)malloc(THREAD_NUM * sizeof(pthread_t)); // ���ٶ��߳�

        for (int i = 0; i < THREAD_NUM; i++) {
            pthread_create(&pthread_set[i], NULL, thread_Numbers_Add, (void *)(intptr_t)i);
        }
        gettimeofday(&start, NULL);

        // �ȴ������߳����
        for (int i = 0; i < THREAD_NUM; i++) {
            pthread_join(pthread_set[i], NULL);
            Sum += Local_sum[i];
            Local_sum[i] = 0;
        }

        gettimeofday(&end, NULL);

        double time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec); // ΢��
        time_use /= 1000000;
        printf("Index: %d, Sum: %.2f, Time: %.8f\n", ori - Test_Rnd, Sum, time_use);
        totle_time += time_use;
        free(pthread_set);
        Sum = 0;
        Global_index = 0; //�ָ�����Ϊ0
        }

    double avg_time = totle_time/ori;
    printf("Continuous index of %d use avg time: %.8f \n", MAX_ELEMENTS, avg_time);
    free(Numbers);
    pthread_mutex_destroy(&mutex); // ���ٻ�����
    return 0; 
}
