#ifndef COMMON_H
#define COMMON_H

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr size_t LOOP_COUNT = 2;

cl::sycl::queue initgpu(int deviceIndex = -1);
void vecadd_kernel(cl::sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int thread_id, int iteration);
void kernel_submission(cl::sycl::queue queue, int thread_id, size_t X);

#endif // COMMON_H