#include <CL/sycl.hpp>
#include <iostream>
#include <thread>
#include "common.h"

int main() {
    try {
        cl::sycl::queue queue1 = initgpu();
        cl::sycl::queue queue2 = initgpu();

        std::thread thread1(kernel_submission_1, queue1, 1, 40000000);
        std::thread thread2(kernel_submission_1, queue2, 2, 40000);

        thread1.join();
        thread2.join();

        std::cout << "All threads have finished execution.\n";
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught in main: " << e.what() << std::endl;
    }

    return 0;
}