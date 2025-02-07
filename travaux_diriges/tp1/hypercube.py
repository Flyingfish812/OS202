from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
token = None

if rank == 0:
    token = 42
    print(f"Process {rank} initializes the token with value {token}")

d = size.bit_length() - 1  # Dimension of the hypercube

# Diffusion of the token in the hypercube
for i in range(d):
    partner = rank ^ (1 << i)
    if rank < partner:
        if token is not None:
            comm.send(token, dest=partner)
    else:
        token = comm.recv(source=partner)
    comm.Barrier()

print(f"Process {rank} received token: {token}")

MPI.Finalize()
