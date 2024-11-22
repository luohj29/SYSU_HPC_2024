#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
 
    if (argc != 4) {
        printf("Usage: %s <M_SIZE> <N_SIZE> <K_SIZE>\n", argv[0]);
        return 1;
    }


    int M_SIZE = atoi(argv[1]);
    int N_SIZE = atoi(argv[2]);
    int K_SIZE = atoi(argv[3]);


    double *a = (double *)malloc(M_SIZE * N_SIZE * sizeof(double));
    double *b = (double *)malloc(N_SIZE * K_SIZE * sizeof(double));
    double *c = (double *)malloc(M_SIZE * K_SIZE * sizeof(double));

 
    for (int i = 0; i < M_SIZE * N_SIZE; i++) {
        a[i] = (double)rand() / RAND_MAX * 10000.0;
    }
    for (int i = 0; i < N_SIZE * K_SIZE; i++) {
        b[i] = (double)rand() / RAND_MAX * 10000.0;
    }
    for (int i = 0; i < M_SIZE * K_SIZE; i++) {
        c[i] = 0.0;
    }


    clock_t start, end;
    start = clock();
    //定义i,k,j顺序的矩阵乘法
    for (int i = 0; i < M_SIZE; i++) {
            for (int k = 0; k < N_SIZE; k++)  {
                for (int j = 0; j < K_SIZE; j++){
                    c[i * K_SIZE + j] += a[i * N_SIZE + k] * b[k * K_SIZE + j];
                }
            }
        }

    // for (int i = 0; i < M_SIZE; i++) {
    //     for (int j = 0; j < K_SIZE; j++){
    //         for (int k = 0; k < N_SIZE; k++){  
    //             c[i * K_SIZE + j] += a[i * N_SIZE + k] * b[k * K_SIZE + j];
    //         }
    //     }
    // }


    end = clock();


    printf("C Matrix Multiplying of %d*%d and %d*%d use time of %f seconds\n", M_SIZE, N_SIZE, N_SIZE, K_SIZE, (double)(end - start) / CLOCKS_PER_SEC);


    free(a);
    free(b);
    free(c);

    return 0;
}
