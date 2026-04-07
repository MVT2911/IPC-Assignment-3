Compile Command : mpicc -fopenmp integration.c -lm -o integration

Execution Commands : 

export OMP_NUM_THREADS=T
mpirun -np P integration FI M tol

T: No of threads
P: number of MPI processes
FI: function ID (0, 1, or 2)
M: execution mode (0, 1 or 2)
tol: error tolerance (e.g., 1e-6, 1e-8)


Machine Specifications : No of cores in PC - 4 


MPI Implementation


mvt@LAPTOP-UIG19S0L:/mnt/c/Users/mvara$ mpicc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.


mvt@LAPTOP-UIG19S0L:/mnt/c/Users/mvara$ mpirun --version
mpirun (Open MPI) 4.1.6
