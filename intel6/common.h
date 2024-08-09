#ifndef COMMON_H
#define COMMON_H

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr size_t N_LOOP = 40000000;
constexpr size_t N_SINGLE = 40000000;
constexpr size_t LOOP_COUNT = 10;

cl::sycl::queue initgpu();
void vecadd_kernel(cl::sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int thread_id, int iteration);
void kernel_submission_1(cl::sycl::queue queue, int thread_id, size_t X);
void kernel_submission_2(cl::sycl::queue queue, int thread_id);

#endif  // COMMON_H