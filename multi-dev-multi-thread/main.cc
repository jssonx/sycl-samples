#include <CL/sycl.hpp>
#include <iostream>
#include <thread>
#include <omp.h>
#include "common.h"

int main()
{
    try
    {
        cl::sycl::queue queue1 = initgpu(0);
        cl::sycl::queue queue2 = initgpu(1);
        cl::sycl::queue queue3 = initgpu(1);

#pragma omp parallel num_threads(3)
        {
#pragma omp sections nowait
            {
#pragma omp section
                kernel_submission(queue1, 1, 10000000);

#pragma omp section
                kernel_submission(queue2, 2, 10000000);

#pragma omp section
                kernel_submission(queue3, 2, 10000000);
            }
        }

        std::cout << "All threads have finished execution.\n";
    }
    catch (cl::sycl::exception const &e)
    {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    return 0;
}