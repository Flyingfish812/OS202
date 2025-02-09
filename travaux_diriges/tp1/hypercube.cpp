#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size, dimension, token = 100; // 初始化 token 为 100

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    dimension = (int)log2(size);
    if (pow(2, dimension) != size) {
        if (rank == 0) {
            printf("Error: Number of processes must be a power of 2.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rank == 0) {
        printf("Process %d initializes token = %d\n", rank, token);
    }

    for (int d = 0; d < dimension; d++) {
        int partner = rank ^ (1 << d);

        if (rank < partner) {
            MPI_Send(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("Process %d received token = %d from process %d at dimension %d\n", rank, token, partner, d);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
