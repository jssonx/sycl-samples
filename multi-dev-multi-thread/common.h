#ifndef COMMON_H
#define COMMON_H

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

constexpr size_t LOOP_COUNT = 10;

std::vector<sycl::device> initgpu();
sycl::queue createQueue(const sycl::device& device);
void vecadd_kernel(sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int iteration, const std::string& func_name);
void vecadd_kernel2(sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int iteration, const std::string& func_name);
void kernel_submission(sycl::queue queue, size_t X, const std::string& func_name);
void kernel_submission2(sycl::queue queue, size_t X, const std::string& func_name);

#endif // COMMON_H