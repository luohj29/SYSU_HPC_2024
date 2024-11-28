#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

pthread_mutex_t shared_var;

void print_matrix(double *matrix, int start_row, int end_row, int columns) {
    if (!matrix || start_row < 0 || end_row <= start_row || columns <= 0) {
        printf("Invalid input parameters\n");
        return;
    }

    printf("Matrix rows %d to %d (columns: %d):\n", start_row, end_row, columns);
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < columns; j++) {
            printf("%8.2f ", matrix[i * columns + j]);
        }
        printf("\n");
    }
}

int M, N;

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (argc != 3) {
        printf("Usage: ./program M N\n");
        MPI_Finalize();
        return 0;
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);

    int avg_rows = M / (size - 1);
    int signal;
    MPI_Status status;

    if (rank == 0) {
        double diff, epsilon = 0.001, mean = 0.0;
        int iterations = 0, print_interval = 1;
        double *current = (double *)malloc(M * N * sizeof(double));
        double *previous = (double *)malloc(M * N * sizeof(double));

        if (!current || !previous) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
// initialize th matirx edge and get the mean number
//********************  
        for (int i = 1; i < M - 1; i++) {
            current[i * N] = 100.0;
            current[i * N + N - 1] = 100.0;
        }
        for (int j = 0; j < N; j++) {
            current[(M - 1) * N + j] = 100.0;
            current[j] = 0.0;
        }

        for (int i = 1; i < M - 1; i++) {
            mean += current[i * N] + current[i * N + N - 1];
        }
        for (int j = 0; j < N; j++) {
            mean += current[(M - 1) * N + j] + current[j];
        }
        mean /= (2 * M + 2 * N - 4);

        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                current[i * N + j] = mean;
            }
        }
//********************

        diff = epsilon;
        double local_diff[size - 1];

        while (diff >= epsilon) {
            diff = 0.0;
            for (int i = 0; i < size - 1; i++) {
                int start_row = i * avg_rows - 1 > 0 ? i * avg_rows - 1 : 0;
                int end_row = (i + 1 == size - 1) ? M : (i + 1) * avg_rows + 1;

                MPI_Send(&end_row, 1, MPI_INT, i + 1, 10, comm);
                MPI_Send(&current[start_row * N], (end_row - start_row) * N, MPI_DOUBLE, i + 1, 1, comm);
            }

            for (int i = 0; i < size - 1; i++) {
                int start_row = i * avg_rows - 1 > 0 ? i * avg_rows - 1 : 0;
                int end_row = (i + 1 == size - 1) ? M : (i + 1) * avg_rows + 1;

                MPI_Recv(&local_diff[i], 1, MPI_DOUBLE, i + 1, 3, comm, &status);
                MPI_Recv(&current[(start_row + 1) * N], (end_row - start_row - 2) * N, MPI_DOUBLE, i + 1, 4, comm, &status);

                diff = fmax(local_diff[i], diff);
            }

            iterations++;
            if (iterations == print_interval) {
                printf("Iteration %d: diff = %f\n", iterations, diff);
                print_interval *= 2;
            }
        }

        printf("Converged after %d iterations.\n", iterations);
        signal = -1;
        for (int i = 0; i < size - 1; i++) {
            MPI_Send(&signal, 1, MPI_INT, i + 1, 10, comm);
        }

        free(current);
        free(previous);
    } else {
        int start_row = rank == 1 ? 1 : avg_rows * (rank - 1);
        int end_row = rank == size - 1 ? M - 1 : avg_rows * rank;

        int extended_start = start_row - 1;
        int extended_end = end_row + 1;

        double *local_current = (double *)malloc((extended_end - extended_start) * N * sizeof(double));
        double *local_previous = (double *)malloc((extended_end - extended_start) * N * sizeof(double));

        if (!local_current || !local_previous) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        double max_diff;

        while (1) {
            MPI_Recv(&signal, 1, MPI_INT, 0, 10, comm, &status);
            if (signal == -1) break;

            MPI_Recv(local_previous, (extended_end - extended_start) * N, MPI_DOUBLE, 0, 1, comm, &status);

            max_diff = 0.0;
            for (int i = 1; i < (extended_end - extended_start) - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    local_current[i * N + j] = 0.25 * (local_previous[(i - 1) * N + j] +
                                                       local_previous[(i + 1) * N + j] +
                                                       local_previous[i * N + j - 1] +
                                                       local_previous[i * N + j + 1]);

                    double diff = fabs(local_current[i * N + j] - local_previous[i * N + j]);
                    if (diff > max_diff) max_diff = diff;
                }
            }

            MPI_Send(&max_diff, 1, MPI_DOUBLE, 0, 3, comm);
            MPI_Send(&local_current[N], (end_row - start_row) * N, MPI_DOUBLE, 0, 4, comm);
        }

        free(local_current);
        free(local_previous);
    }

    MPI_Finalize();
    return 0;
}
