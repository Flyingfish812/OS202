from mpi4py import MPI
import numpy as np

def monte_carlo_pi(n_points):
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    inside_circle = np.sum(x**2 + y**2 <= 1)
    return inside_circle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_total = 10**6
n_local = n_total // size

inside_local = monte_carlo_pi(n_local)

inside_global = comm.reduce(inside_local, op=MPI.SUM, root=0)

if rank == 0:
    pi_estimate = 4 * inside_global / n_total
    print(f"Approximated Pi: {pi_estimate}")

MPI.Finalize()
