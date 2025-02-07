from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Size
dim = 120
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
u = np.array([i+1. for i in range(dim)])

# Line range for current process
rows_per_proc = dim // size
remainder = dim % size

if rank < remainder:
    start_row = rank * (rows_per_proc + 1)
    end_row = start_row + rows_per_proc + 1
else:
    start_row = rank * rows_per_proc + remainder
    end_row = start_row + rows_per_proc

A_local = A[start_row:end_row, :]
u_bcast = comm.bcast(u, root=0)

start_time = MPI.Wtime()

v_local = A_local @ u_bcast

if rank == 0:
    v_result = np.empty(dim, dtype=np.double)
else:
    v_result = None

sendcounts = [rows_per_proc + 1 if i < remainder else rows_per_proc for i in range(size)]
displacements = [sum(sendcounts[:i]) for i in range(size)]

comm.Gatherv(sendbuf=v_local, recvbuf=(v_result, sendcounts, displacements, MPI.DOUBLE), root=0)

end_time = MPI.Wtime()
elapsed_time = end_time - start_time

# Process 0 prints the final result and execution time
if rank == 0:
    print(f"Final result v = {v_result}")
    print(f"{size} processes with Execution Time = {elapsed_time:.6f} seconds")
