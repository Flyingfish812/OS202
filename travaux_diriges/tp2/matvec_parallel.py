from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix size
dim = 120
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
u = np.array([i+1. for i in range(dim)])

# Column range for current process
cols_per_proc = dim // size
remainder = dim % size

if rank < remainder:
    start_col = rank * (cols_per_proc + 1)
    end_col = start_col + cols_per_proc + 1
else:
    start_col = rank * cols_per_proc + remainder
    end_col = start_col + cols_per_proc

# Sub-matrix for current process
A_local = A[:, start_col:end_col]
u_local = u[start_col:end_col]

start_time = MPI.Wtime()

v_local = A_local @ u_local

v_result = np.zeros(dim) if rank == 0 else None
comm.Reduce(v_local, v_result, op=MPI.SUM, root=0)

end_time = MPI.Wtime()
elapsed_time = end_time - start_time

# Process 0 print the result and the time
if rank == 0:
    print(f"Final result v = {v_result}")
    print(f"Process {rank} with Execution Time = {elapsed_time:.6f} seconds")
else:
    print(f"Process {rank} with Execution Time = {elapsed_time:.6f} seconds")
