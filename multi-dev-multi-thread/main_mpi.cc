#include <sycl/sycl.hpp>
#include <iostream>
#include <mpi.h>
#include "common.h"

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try
    {
        // Each rank queries its available devices
        std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);

        if (devices.empty()) {
            std::cerr << "Error: No devices found for rank " << rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Each rank will work with the first available device (as limited by ZE_AFFINITY_MASK)
        sycl::queue queue = createQueue(devices[0]);

        std::cout << "Rank " << rank << " is using device: " 
                  << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

        // Perform kernel execution on the rank-specific device
        if (rank == 0) {
            kernel_submission2(queue, 10000000, "kernel" + std::to_string(rank));
        } else {
            kernel_submission(queue, 10000000, "kernel" + std::to_string(rank));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "All ranks have finished execution.\n";
        }
    }
    catch (sycl::exception const &e)
    {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
