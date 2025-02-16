from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 100000
# NLoc = N // size + (1 if rank < (N % size) else 0)

def scatter_index(number, quantiles):
    for i in range(len(quantiles) - 1):
        if quantiles[i] <= number < quantiles[i + 1] or (i == len(quantiles) - 2 and number == quantiles[i + 1]):
            return i
        
# Data generation and distribution
if rank == 0:
    np.random.seed(42)
    data = np.random.randint(-32768, 32767, size=N)
    with open(f'output.txt', 'w') as out:
        out.write(f"Valeurs initiales: {data}\n")
    start_time = time.time()

    buckets = {i: [] for i in range(size)}
    quantiles = quantiles = np.quantile(data, np.linspace(0, 1, size + 1))
    for number in data:
        bucket_index = scatter_index(number, quantiles)
        buckets[bucket_index].append(number)
else:
    buckets = None

# Scatter, local sort, and gather
local_data = comm.scatter([buckets[i] if buckets else [] for i in range(size)], root=0)
local_sorted = sorted(local_data)
sorted_data = comm.gather(local_sorted, root=0)

# Process 0 concatenates the sorted data
if rank == 0:
    final_sorted_data = [item for sublist in sorted_data for item in sublist]
    final_sorted_data = np.array(final_sorted_data)
    end_time = time.time()
    with open(f'output.txt', 'a') as out:
        out.write(f"Valeurs finales: {final_sorted_data}\n")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print(f"Number of buckets used: {size}")
