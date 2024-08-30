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

#if 0
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
#endif

#if 0
#pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
            switch(thread_num) {
                case 0:
                    kernel_submission_1(queue1, 1, 40000000);
                    break;
                case 1:
                    kernel_submission_1(queue2, 2, 40000000);
                    break;
                case 2:
                    kernel_submission_1(queue3, 2, 400000);
                    break;
            }
        }
#endif

#if 1
        std::thread thread1(kernel_submission_1, queue1, 1, 40000000);
        std::thread thread2(kernel_submission_1, queue2, 2, 40000000);
        std::thread thread3(kernel_submission_1, queue3, 2, 400000);

        thread1.join();
        thread2.join();
        thread3.join();
#endif

        std::cout << "All threads have finished execution.\n";
    }
    catch (cl::sycl::exception const &e)
    {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    return 0;
}