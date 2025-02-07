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

# Process 0 for complexity calculation
if rank == 0:
    complexity = np.zeros(height)
    for y in range(height):
        row_complexity = 0
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            row_complexity += mandelbrot_set.convergence(c, smooth=False)
        complexity[y] = row_complexity / width

    # Divide the mission
    total_work = np.sum(complexity)
    work_per_process = total_work / size
    start_indices = [0]
    accumulated_work = 0

    for y in range(height):
        accumulated_work += complexity[y]
        if accumulated_work >= (len(start_indices) * work_per_process):
            start_indices.append(y)

    start_indices.append(height)

    # sendcounts and displacements
    sendcounts = [(start_indices[i+1] - start_indices[i]) * width for i in range(size)]
    displacements = [sum(sendcounts[:i]) for i in range(size)]

    # Distribute the tasks
    tasks = [start_indices[i:i+2] for i in range(size)]
else:
    tasks = None
    sendcounts = None
    displacements = None

# Broadcast the tasks
tasks = comm.scatter(tasks, root=0)
sendcounts = comm.bcast(sendcounts, root=0)
displacements = comm.bcast(displacements, root=0)

start_row, end_row = tasks

# Mandelbrot
local_height = end_row - start_row
local_convergence = np.empty((local_height, width), dtype=np.double)

start_time = time.time()
for j, y in enumerate(range(start_row, end_row)):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_convergence[j, x] = mandelbrot_set.convergence(c, smooth=True)

end_time = time.time()
print(f"Process {rank} finished in {end_time - start_time:.4f} seconds.")

# Process 0 pre-distribute final_convergence
if rank == 0:
    final_convergence = np.empty((height, width), dtype=np.double)
else:
    final_convergence = np.empty(1, dtype=np.double)

comm.Gatherv(
    sendbuf=local_convergence.flatten(),
    recvbuf=(final_convergence if rank == 0 else None, sendcounts, displacements, MPI.DOUBLE),
    root=0
)

if rank == 0:
    final_convergence = final_convergence.reshape(height, width)
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence) * 255))
    image.save("mandelbrot_optimized_2.png")
    print(f"Saved Mandelbrot image as mandelbrot_optimized_2.png")
