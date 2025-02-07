from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    token = 1
    print(f"Process {rank} initializes the token with value {token}")
    comm.send(token, dest=1)
    token = comm.recv(source=size - 1)
    print(f"Process {rank} received final token: {token}")
else:
    token = comm.recv(source=rank - 1)
    token += 1
    comm.send(token, dest=(rank + 1) % size)

MPI.Finalize()