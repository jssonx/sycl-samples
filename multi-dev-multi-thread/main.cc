#include <sycl/sycl.hpp>
#include <iostream>
#include <thread>
#include <omp.h>
#include "common.h"


int main()
{
    try
    {
        std::vector<sycl::device> devices = initgpu();
        sycl::queue queue1 = createQueue(devices[0]);
        sycl::queue queue2 = createQueue(devices[1]);
        sycl::queue queue3 = createQueue(devices[2]);
        sycl::queue queue4 = createQueue(devices[3]);

        #pragma omp parallel num_threads(4)
        {
            #pragma omp sections nowait
            {
                #pragma omp section
                kernel_submission(queue1, 10000000, "kernel1"); // queue, X, func_name

                #pragma omp section
                kernel_submission(queue2, 10000000, "kernel2");

                #pragma omp section
                kernel_submission(queue3, 10000000, "kernel3");

                #pragma omp section
                kernel_submission(queue4, 10000000, "kernel4");
            }
        }

        std::cout << "All threads have finished execution.\n";
    }
    catch (sycl::exception const &e)
    {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    return 0;
}
