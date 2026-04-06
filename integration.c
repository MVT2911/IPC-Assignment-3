#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Message Tags */
#define TAG_WORK        1
#define TAG_RESULT      2
#define TAG_NEW_TASK    3
#define TAG_STOP        4
#define TAG_WORK_REQ    5

/* Function Prototypes */
double get_func(int id, double x);
double simpson(int id, double a, double b);
double adaptive_simpson_serial(int id, double a, double b, double tol, double whole, int *intervals);
double adaptive_simpson_hybrid(int id, double a, double b, double tol, double whole, int *intervals);

/* Supported Functions */
double get_func(int id, double x) {
    switch(id) {
        case 0: return sin(x) + 0.5 * cos(3 * x);
        case 1: return 1.0 / (1.0 + 100.0 * pow(x - 0.3, 2));
        case 2: return sin(200 * x) * exp(-x);
        default: return 0;
    }
}

double simpson(int id, double a, double b) {
    double c = (a + b) / 2.0;
    return (fabs(b - a) / 6.0) * (get_func(id, a) + 4.0 * get_func(id, c) + get_func(id, b));
}

/* Mode 0: Serial */
double adaptive_simpson_serial(int id, double a, double b, double tol, double whole, int *intervals) {
    (*intervals)++;
    double m = (a + b) / 2.0;
    double left = simpson(id, a, m);
    double right = simpson(id, m, b);
    if (fabs(whole - (left + right)) <= 15 * tol) {
        return left + right + (whole - (left + right)) / 15.0;
    }
    return adaptive_simpson_serial(id, a, m, tol / 2.0, left, intervals) + 
           adaptive_simpson_serial(id, m, b, tol / 2.0, right, intervals);
}

/* Mode 2: Hybrid OpenMP Tasks */
double adaptive_simpson_hybrid(int id, double a, double b, double tol, double whole, int *intervals) {
    #pragma omp atomic
    (*intervals)++;

    double m = (a + b) / 2.0;
    double left_s = simpson(id, a, m);
    double right_s = simpson(id, m, b);

    if (fabs(whole - (left_s + right_s)) <= 15 * tol) {
        return left_s + right_s + (whole - (left_s + right_s)) / 15.0;
    }

    double left_res, right_res;
    #pragma omp task shared(left_res)
    left_res = adaptive_simpson_hybrid(id, a, m, tol / 2.0, left_s, intervals);

    #pragma omp task shared(right_res)
    right_res = adaptive_simpson_hybrid(id, m, b, tol / 2.0, right_s, intervals);

    #pragma omp taskwait
    return left_res + right_res;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Usage: mpirun -np P ./integration func_id mode tol\n");
        MPI_Finalize();
        return 0;
    }

    int func_id = atoi(argv[1]);
    int mode = atoi(argv[2]);   
    double tol = atof(argv[3]); 
    int local_intervals = 0;
    double final_integral = 0;
    int total_intervals = 0;

    double start_time = MPI_Wtime();

    /* MODE 0: SERIAL */
    if (mode == 0 && rank == 0) {
        final_integral = adaptive_simpson_serial(func_id, 0, 1, tol, simpson(func_id, 0, 1), &local_intervals);
        total_intervals = local_intervals;
    } 

    /* MODE 1: MPI DYNAMIC */
    else if (mode == 1) {
        if (rank == 0) {
            int active_workers = 0;
            double stack[10000][2]; 
            int top = -1;
            stack[++top][0] = 0.0; stack[top][1] = 1.0;

            while (top >= 0 || active_workers > 0) {
                MPI_Status status;
                double msg[3];
                MPI_Recv(msg, 3, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == TAG_WORK_REQ && top >= 0) {
                    MPI_Send(stack[top--], 2, MPI_DOUBLE, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
                    active_workers++;
                } else if (status.MPI_TAG == TAG_RESULT) {
                    final_integral += msg[0];
                    local_intervals += (int)msg[1];
                    active_workers--;
                } else if (status.MPI_TAG == TAG_NEW_TASK) {
                    stack[++top][0] = msg[0]; stack[top][1] = msg[1];
                }
            }
            for (int i = 1; i < size; i++) MPI_Send(NULL, 0, MPI_DOUBLE, i, TAG_STOP, MPI_COMM_WORLD);
            total_intervals = local_intervals;
        } else {
            while (1) {
                double range[2]; MPI_Status status;
                MPI_Send(NULL, 0, MPI_DOUBLE, 0, TAG_WORK_REQ, MPI_COMM_WORLD);
                MPI_Recv(range, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_STOP) break;

                double m = (range[0] + range[1]) / 2.0;
                double w = simpson(func_id, range[0], range[1]);
                double l = simpson(func_id, range[0], m), r = simpson(func_id, m, range[1]);
                
                int intervals_counted = 1;
                if (fabs(w - (l + r)) <= 15 * tol) {
                    double res[2] = {l + r + (w - (l + r)) / 15.0, (double)intervals_counted};
                    MPI_Send(res, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
                } else {
                    double t1[2] = {m, range[1]}, t2[2] = {range[0], m};
                    MPI_Send(t1, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD);
                    MPI_Send(t2, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD);
                    double res[2] = {0, (double)intervals_counted};
                    MPI_Send(res, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
                }
            }
        }
    }

    /* MODE 2: HYBRID */
    else if (mode == 2) {
        double hybrid_local_sum = 0;
        if (rank == 0) {
            int K = 100; // Static distribution
            for (int i = 0; i < K; i++) {
                double range[2] = {(double)i/K, (double)(i+1)/K};
                int dest = (i % (size - 1)) + 1; 
                MPI_Send(range, 2, MPI_DOUBLE, dest, TAG_WORK, MPI_COMM_WORLD);
            }
            for (int i = 1; i < size; i++) MPI_Send(NULL, 0, MPI_DOUBLE, i, TAG_STOP, MPI_COMM_WORLD);
        } else {
            double range[2]; MPI_Status status;
            while (1) {
                MPI_Recv(range, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_STOP) break;
                #pragma omp parallel
                {
                    #pragma omp single
                    hybrid_local_sum += adaptive_simpson_hybrid(func_id, range[0], range[1], tol, simpson(func_id, range[0], range[1]), &local_intervals);
                }
            }
        }
        MPI_Reduce(&hybrid_local_sum, &final_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_intervals, &total_intervals, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("\n--- Results (Mode %d) ---\n", mode);
        printf("Integral: %.10f\n", final_integral);
        printf("Total Intervals: %d\n", total_intervals); 
        printf("Function ID: %d\n", func_id);
        printf("Error Tolerance: %.10f\n", tol);
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}