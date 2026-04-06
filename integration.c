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
double adaptive_simpson_serial(int id, double a, double b, double tol, double whole);
double adaptive_simpson_hybrid(int id, double a, double b, double tol, double whole);

/* Numerical Integration Functions */
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

/* Mode 0: Serial Implementation */
double adaptive_simpson_serial(int id, double a, double b, double tol, double whole) {
    double m = (a + b) / 2.0;
    double left = simpson(id, a, m);
    double right = simpson(id, m, b);
    if (fabs(whole - (left + right)) <= 15 * tol) { 
        return left + right + (whole - (left + right)) / 15.0;
    }
    return adaptive_simpson_serial(id, a, m, tol / 2.0, left) + 
           adaptive_simpson_serial(id, m, b, tol / 2.0, right); 
}

/* Mode 2: Hybrid OpenMP Task Implementation */
double adaptive_simpson_hybrid(int id, double a, double b, double tol, double whole) {
    double m = (a + b) / 2.0;
    double left_s = simpson(id, a, m);
    double right_s = simpson(id, m, b);

    if (fabs(whole - (left_s + right_s)) <= 15 * tol) {
        return left_s + right_s + (whole - (left_s + right_s)) / 15.0;
    }

    double left_res, right_res;
    
    /* Parallelize recursion using OpenMP tasks */
    #pragma omp task shared(left_res) 
    left_res = adaptive_simpson_hybrid(id, a, m, tol / 2.0, left_s);

    #pragma omp task shared(right_res)
    right_res = adaptive_simpson_hybrid(id, m, b, tol / 2.0, right_s);

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

    double start_time = MPI_Wtime(); 

    /* MODE 0: SERIAL BASELINE */
    if (mode == 0 && rank == 0) {
        double res = adaptive_simpson_serial(func_id, 0, 1, tol, simpson(func_id, 0, 1)); 
        printf("Result: %.12f, Time: %f\n", res, MPI_Wtime() - start_time);
    }

/* MODE 1: MPI DYNAMIC MASTER/WORKER */
    else if (mode == 1) {
        if (rank == 0) {
            double total_integral = 0;
            int active_workers = 0;
            double stack[2000][2]; /
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
                    total_integral += msg[0];
                    active_workers--;
                } else if (status.MPI_TAG == TAG_NEW_TASK) { 
                    stack[++top][0] = msg[0]; stack[top][1] = msg[1];
                }
            }
            for (int i = 1; i < size; i++) MPI_Send(NULL, 0, MPI_DOUBLE, i, TAG_STOP, MPI_COMM_WORLD);
            printf("Mode 1 Result: %.12f, Time: %f\n", total_integral, MPI_Wtime() - start_time);
        } else {
            /* Worker logic for Mode 1 */
            while (1) {
                double range[2]; MPI_Status status;
                MPI_Send(NULL, 0, MPI_DOUBLE, 0, TAG_WORK_REQ, MPI_COMM_WORLD); 
                MPI_Recv(range, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                if (status.MPI_TAG == TAG_STOP) break;

                double m = (range[0] + range[1]) / 2.0;
                double w = simpson(func_id, range[0], range[1]);
                double l = simpson(func_id, range[0], m), r = simpson(func_id, m, range[1]);

                if (fabs(w - (l + r)) <= 15 * tol) {
                    double res = l + r + (w - (l + r)) / 15.0;
                    MPI_Send(&res, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
                } else {
                    double task[2] = {m, range[1]};
                    MPI_Send(task, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD); 
                    /* Process other half locally or send back */
                    double task2[2] = {range[0], m};
                    MPI_Send(task2, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD);
                    double dummy = 0; MPI_Send(&dummy, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
                }
            }
        }
    }

    // --- MODE 2: DYNAMIC MASTER/WORKER --- 
    else if (mode == 2) {
        if (rank == 0) { // MASTER 
            Task stack[20000]; 
            int top = 0;
            int active_workers = 0;

            // Initial task [0, 1] 
            stack[top++] = (Task){0.0, 1.0};

            while (top > 0 || active_workers > 0) { // Termination detection logic 
                MPI_Status status;
                double msg_buffer[2]; 
                MPI_Recv(msg_buffer, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                int worker = status.MPI_SOURCE;

                if (status.MPI_TAG == TAG_WORK_REQ) {
                    if (top > 0) {
                        Task t = stack[--top];
                        MPI_Send(&t, sizeof(Task), MPI_BYTE, worker, TAG_WORK_TASK, MPI_COMM_WORLD);
                        active_workers++;
                    } else {
                        // Queue empty, but other workers might still be generating tasks
                        MPI_Send(NULL, 0, MPI_BYTE, worker, TAG_TERMINATE, MPI_COMM_WORLD);
                    }
                } 
                else if (status.MPI_TAG == TAG_RESULT) {
                    total_integral += msg_buffer[0];
                    total_intervals += (int)msg_buffer[1];
                    active_workers--;
                }
                else if (status.MPI_TAG == TAG_NEW_TASK) {
                    stack[top++] = (Task){msg_buffer[0], msg_buffer[1]};
                }
            }
            // Signal all workers to shut down permanently 
            for (int i = 1; i < size; i++) {
                MPI_Send(NULL, 0, MPI_BYTE, i, TAG_TERMINATE, MPI_COMM_WORLD);
            }
        }
        else { // WORKER 
            while (1) {
                MPI_Send(NULL, 0, MPI_BYTE, 0, TAG_WORK_REQ, MPI_COMM_WORLD); // 1. Request work 
                
                Task t;
                MPI_Status status;
                MPI_Recv(&t, sizeof(Task), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // 2. Receive 

                if (status.MPI_TAG == TAG_TERMINATE) break; // End signal 

                // 3. Apply adaptive logic 
                double m = (t.L + t.R) / 2.0;
                double left_s = simpson(func_id, t.L, m);
                double right_s = simpson(func_id, m, t.R);
                double whole_s = simpson(func_id, t.L, t.R);

                if (fabs(left_s + right_s - whole_s) <= 15 * tol) {
                    // Accepted result 
                    double res[2] = {left_s + right_s + (left_s + right_s - whole_s) / 15.0, 1.0};
                    MPI_Send(res, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD); 
                } else {
                    // Split 
                    double new_task[2] = {m, t.R}; 
                    MPI_Send(new_task, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD); // Send half back 
                    
                    // Worker handles the other half immediately to stay busy 
                    // (Simplified logic: push back to master for this assignment version)
                    double left_task[2] = {t.L, m};
                    MPI_Send(left_task, 2, MPI_DOUBLE, 0, TAG_NEW_TASK, MPI_COMM_WORLD);
                    
                    // Reset active_worker count for this worker since it's "done" with the original task
                    double zero_res[2] = {0, 0};
                    MPI_Send(zero_res, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
                }
            }
        }
    }

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n--- Results (Mode %d) ---\n", mode);
        printf("Integral: %.10f\n", total_integral);
        printf("Total Intervals: %d\n", total_intervals); // Deterministic work indicator 
        printf("Function ID: %d\n", func_id);
        printf("Error Tolerance: %.10f\n", tol);
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}