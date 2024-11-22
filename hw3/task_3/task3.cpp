//g++ -o test test.cpp -lpthread
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
// #define _GNU_SOURCE

int avg_time = 0;
int total_count = 0;
pthread_mutex_t mutex;
int stick_this_thread_to_core(int rank) {
   int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
   int core_id = rank % num_cores; //将线程哈希绑定到一个固定的core
   if (core_id < 0 || core_id >= num_cores)
      return EINVAL;

   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(core_id, &cpuset);

   pthread_t current_thread = pthread_self(); 
   printf("tie the rank %d thread to cpu id %d of total core sum of %d\n", rank+1, core_id, num_cores);   
   return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

void * compute_thread(void *arg){
    int rank = (intptr_t)arg; //the rank of the thread
    stick_this_thread_to_core(rank); 

    printf("%d compute_thread start to work\n", rank+1);
    double x , y;
    int time = avg_time;
    int local_count = 0;
    while(time--){
        x = rand() / (double)RAND_MAX;
        y = rand() / (double)RAND_MAX;
        if (y <= x *x ){
            local_count ++;
        }
    }
    pthread_mutex_lock(&mutex);
    total_count += local_count;
    printf("%d compute_thread finish work with local_count of %d\n", rank+1, local_count);
    pthread_mutex_unlock(&mutex); 

    return NULL;   
}

//the input is thread_num, test_time
int main(int argc, char* argv[]){
    if(argc != 3){
        //need 2 out para
    }
    int thread_num = atoi(argv[1]);
    int total_time = atoi(argv[2]);;
    avg_time = total_time/thread_num;
 
    int i;
    pthread_t *thread_set = NULL;
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_unlock(&mutex);

    thread_set = (pthread_t*)malloc(thread_num * sizeof(pthread_t));

    for (i = 0; i < thread_num; i++) {
        pthread_create(thread_set + i, NULL, compute_thread, (void *)(intptr_t)i);
    }

    for (i = 0; i < thread_num; i++) {
        pthread_join(thread_set[i], NULL);
    }

    double real_area = (double)1/3;
    double comp_area = (double)total_count / total_time;

    printf("Result-------------------------------------\n");
    printf("Running %d round\nGet total count as %d\n", total_time, total_count);
    printf("Area estimated in Monte-carlo method:%lf\n", comp_area);
    printf("Real area:%lf\n", real_area);
    printf("Mistake:%lf\n", real_area - comp_area);

    free(thread_set);   
    return 0; 
}
