from mpi4py import MPI
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
import matplotlib.cm
import time

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

width, height = 1024, 1024
max_iterations = 50

mandelbrot_set = MandelbrotSet(max_iterations=max_iterations, escape_radius=10)
scaleX = 3.0 / width
scaleY = 2.25 / height

if rank == 0:  # Master process
    start_time = time.time()

    task_queue = list(range(height))  # Task list
    result_data = np.empty((height, width), dtype=np.double)
    num_active_workers = size - 1  # Process count

    # Initial task distribution
    for worker in range(1, size):
        if task_queue:
            row = task_queue.pop(0)
            comm.send(row, dest=worker, tag=1)  # Send to worker
        else:
            comm.send(None, dest=worker, tag=0)

    # Dynamic task distribution
    while num_active_workers > 0:
        # Wait for worker to apply for a task
        worker = comm.recv(source=MPI.ANY_SOURCE, tag=3)
        
        # Waiting for worker to return a task
        row, data = comm.recv(source=worker, tag=2)

        # Store the result
        result_data[row, :] = data

        if task_queue:
            # Distribute new tasks
            new_row = task_queue.pop(0)
            comm.send(new_row, dest=worker, tag=1)
        else:
            # No new tasks, signal worker to finish
            comm.send(None, dest=worker, tag=0)
            num_active_workers -= 1

    master_end_time = time.time()
    master_duration = master_end_time - start_time
    print(f"Master process finished in {master_duration:.4f} seconds.")

    # Save the image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(result_data) * 255))
    image.save("mandelbrot_master_slave_fixed.png")
    print("Saved Mandelbrot image as mandelbrot_master_slave_fixed.png")

else:  # Worker process
    total_worker_time = 0

    while True:
        # Apply for a task
        comm.send(rank, dest=0, tag=3)
        
        # Receive a task from the master
        row = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())

        if row is None:
            break

        worker_start_time = time.time()

        row_data = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * row)
            row_data[x] = mandelbrot_set.convergence(c, smooth=True)

        worker_end_time = time.time()
        task_duration = worker_end_time - worker_start_time
        total_worker_time += task_duration

        # Send the result
        comm.send((row, row_data), dest=0, tag=2)

    print(f"Worker {rank} finished in {total_worker_time:.4f} seconds.")