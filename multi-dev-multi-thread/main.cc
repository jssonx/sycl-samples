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
                kernel_submission_1(queue1, 1, 40000000);

#pragma omp section
                kernel_submission_1(queue2, 2, 40000000);

#pragma omp section
                kernel_submission_1(queue3, 2, 400000);
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