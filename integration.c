#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// MPI Tags for communication 
#define TAG_WORK_REQ    1  // Worker requesting work
#define TAG_WORK_TASK   2  // Master sending an interval
#define TAG_RESULT      3  // Worker sending partial integral
#define TAG_NEW_TASK    4  // Worker sending a split sub-interval back
#define TAG_TERMINATE   0  // Master telling worker to stop 

typedef struct {
    double L, R;
} Task;

// Function definitions as per assignment requirements 
double f(int id, double x) {
    switch(id) {
        case 0: return sin(x) + 0.5 * cos(3 * x);
        case 1: return 1.0 / (1.0 + 100.0 * pow(x - 0.3, 2));
        case 2: return sin(200 * x) * exp(-x);
        default: return 0;
    }
}

// Standard Simpson's Rule over [a, b]
double simpson(int id, double a, double b) {
    double m = (a + b) / 2.0;
    return ((b - a) / 6.0) * (f(id, a) + 4 * f(id, m) + f(id, b));
}

// Core Adaptive Logic used by all modes
double adaptive_recursive(int id, double a, double b, double tol, int *count) {
    double m = (a + b) / 2.0;
    double left_s = simpson(id, a, m);
    double right_s = simpson(id, m, b);
    double whole_s = simpson(id, a, b);

    // If the difference is within tolerance, accept 
    if (fabs(left_s + right_s - whole_s) <= 15 * tol) {
        (*count)++;
        return left_s + right_s + (left_s + right_s - whole_s) / 15.0;
    }
    // Otherwise, split and continue 
    return adaptive_recursive(id, a, m, tol/2.0, count) + 
           adaptive_recursive(id, m, b, tol/2.0, count);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Usage: mpirun -np P integration func_id mode tol\n");
        MPI_Finalize();
        return 0;
    }

    int func_id = atoi(argv[1]);
    int mode = atoi(argv[2]);
    double tol = atof(argv[3]);
    double total_integral = 0;
    int total_intervals = 0;
    double start_time, end_time;

    start_time = MPI_Wtime(); 

    // --- MODE 0: SERIAL BASELINE --- 
    if (mode == 0) {
        if (rank == 0) {
            total_integral = adaptive_recursive(func_id, 0, 1, tol, &total_intervals);
        }
    }

    // --- MODE 1: STATIC DECOMPOSITION --- 
    else if (mode == 1) {
        int K = 100; // Chosen value for K coarse intervals 
        double step = 1.0 / K;
        double local_sum = 0;
        int local_count = 0;

        for (int i = rank; i < K; i += size) { // Round-robin distribution 
            local_sum += adaptive_recursive(func_id, i*step, (i+1)*step, tol/K, &local_count);
        }

        MPI_Reduce(&local_sum, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
        MPI_Reduce(&local_count, &total_intervals, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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