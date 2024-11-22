#include <iostream>
#include <pthread.h>
#include <cmath>
#include <cstdlib> // for malloc

int b_pow_2 ;
int minus_4ac;
double sqrt_result;

pthread_mutex_t mutex_b_pow_2; //for b^2
pthread_mutex_t mutex_minus_4ac; //for -4ac
pthread_mutex_t mutex_sqrt_result; //for sqrt result

pthread_cond_t cond_b_pow_2; //for b^2 ready
pthread_cond_t cond_minus_4ac; //for -4ac ready

void* mul_fun(void* arg) {
    int con = *(static_cast<int*>(arg));
    if (con == 1) { // calculate b^2
        pthread_mutex_lock(&mutex_b_pow_2);
        int coef1 = *(static_cast<int*>(arg) + 1);
        b_pow_2 = coef1 * coef1;
        // printf("cond1 win\n");
        pthread_cond_signal(&cond_b_pow_2);
        pthread_mutex_unlock(&mutex_b_pow_2);
    } else { // calculate -4ac
        pthread_mutex_lock(&mutex_minus_4ac);
        int coef1 = *(static_cast<int*>(arg) + 1);
        int coef2 = *(static_cast<int*>(arg) + 2);
        minus_4ac = -4 * coef1 * coef2;
        // printf("cond2 win\n");
        pthread_cond_signal(&cond_minus_4ac);
        pthread_mutex_unlock(&mutex_minus_4ac);
    }
    return NULL;
}

void* sqrt_fun(void* arg) {
    pthread_mutex_lock(&mutex_b_pow_2);
    if(b_pow_2 == 0){
        // printf("con1_make\n");
        pthread_cond_wait(&cond_b_pow_2, &mutex_b_pow_2);
    }

    pthread_mutex_unlock(&mutex_b_pow_2);

    pthread_mutex_lock(&mutex_minus_4ac);
    if(b_pow_2 == 0){
        // printf("con2_make\n");
        pthread_cond_wait(&cond_minus_4ac, &mutex_minus_4ac); 
    } 
    pthread_mutex_unlock(&mutex_minus_4ac);

    // Now it's safe to calculate the square root
    pthread_mutex_lock(&mutex_sqrt_result);
    printf("b_squre: %d, minus_4ac: %d\n", b_pow_2, minus_4ac);
    sqrt_result = sqrt(b_pow_2 + minus_4ac);
    pthread_mutex_unlock(&mutex_sqrt_result);
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Need exactly 3 parameters!\n";
        return 1;
    }
    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = atoi(argv[3]);

    pthread_mutex_init(&mutex_b_pow_2, NULL);
    pthread_mutex_init(&mutex_minus_4ac, NULL);
    pthread_mutex_init(&mutex_sqrt_result, NULL);
    pthread_cond_init(&cond_b_pow_2, NULL);
    pthread_cond_init(&cond_minus_4ac, NULL);

    void* data1 = malloc(2 * sizeof(int));
    void* data2 = malloc(3 * sizeof(int));
    *(static_cast<int*>(data1)) = 1;
    *(static_cast<int*>(data1) + 1) = b;
    *(static_cast<int*>(data2)) = 0;
    *(static_cast<int*>(data2) + 1) = a;
    *(static_cast<int*>(data2) + 2) = c;

    // Allocate memory for pthread_t objects
    pthread_t* threads = (pthread_t*)malloc(2 * sizeof(pthread_t));
    if (threads == NULL) {
        std::cerr << "Error allocating memory for threads\n";
        return 1;
    }

    pthread_create(&threads[0], NULL, mul_fun, data1);
    pthread_create(&threads[1], NULL, mul_fun, data2);
    pthread_create(&threads[2], NULL, sqrt_fun, NULL);

    pthread_join(threads[2], NULL);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    double root1 = (-b + sqrt_result) / (2 * a);
    double root2 = (-b - sqrt_result) / (2 * a);
    std::cout << "Roots: " << root1 << '\n' << root2 << std::endl;

    free(data2);
    free(data1);
    free(threads);

    pthread_mutex_destroy(&mutex_b_pow_2);
    pthread_mutex_destroy(&mutex_minus_4ac);
    pthread_mutex_destroy(&mutex_sqrt_result);
    pthread_cond_destroy(&cond_b_pow_2);
    pthread_cond_destroy(&cond_minus_4ac);

    return 0;
}