#ifndef COMMON_H
#define COMMON_H

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr size_t LOOP_COUNT = 10;

std::vector<cl::sycl::device> initgpu();
cl::sycl::queue createQueue(const cl::sycl::device& device);
void vecadd_kernel(cl::sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int iteration, const std::string& func_name);
void kernel_submission(cl::sycl::queue queue, size_t X, const std::string& func_name);

#endif // COMMON_H