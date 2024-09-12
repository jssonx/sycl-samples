#include <CL/sycl.hpp>
#include <iostream>
#include <mpi.h>
#include "common.h"

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<cl::sycl::device> devices;

    try
    {
        if (rank == 0)
        {
            devices = initgpu();
            int num_devices = devices.size();
            MPI_Bcast(&num_devices, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
        {
            int num_devices;
            MPI_Bcast(&num_devices, 1, MPI_INT, 0, MPI_COMM_WORLD);
            devices.resize(num_devices);
        }

        if (rank >= devices.size()) {
            std::cerr << "Error: Not enough devices for rank " << rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        cl::sycl::queue queue = createQueue(devices[rank]);
        kernel_submission(queue, 10000000, "kernel" + std::to_string(rank));

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "All ranks have finished execution.\n";
        }
    }
    catch (cl::sycl::exception const &e)
    {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}