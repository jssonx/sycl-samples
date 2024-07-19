#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>

using namespace sycl;

// Array size for this example.
size_t array_size = 100000000;
constexpr int ITERATIONS = 100;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum".
//************************************
void VectorAdd(queue &q, const int *a, const int *b, int *sum, size_t size) {
  range<1> num_items{size};
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
  e.wait();
}

//************************************
// Initialize the array from 0 to array_size - 1
//************************************
void InitializeArray(int *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char *argv[]) {
  auto total_start_time = std::chrono::high_resolution_clock::now();

  if (argc > 1) array_size = std::stoi(argv[1]);

  auto selector = default_selector_v;

  try {
    queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    int *a = malloc_shared<int>(array_size, q);
    int *b = malloc_shared<int>(array_size, q);
    int *sum_sequential = malloc_shared<int>(array_size, q);
    int *sum_parallel = malloc_shared<int>(array_size, q);

    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr)) {
      if (a != nullptr) free(a, q);
      if (b != nullptr) free(b, q);
      if (sum_sequential != nullptr) free(sum_sequential, q);
      if (sum_parallel != nullptr) free(sum_parallel, q);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, array_size);
    InitializeArray(b, array_size);

    // Compute the sum of two arrays in sequential for validation.
    for (size_t i = 0; i < array_size; i++) sum_sequential[i] = a[i] + b[i];

    // Vector addition in SYCL.
    auto kernel_start_time = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS; iter++) {
      VectorAdd(q, a, b, sum_parallel, array_size);

      if (iter % 10 == 0) {
        std::cout << "Completed iteration " << iter << std::endl;
      }
    }
    auto kernel_end_time = std::chrono::high_resolution_clock::now();
    auto kernel_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            kernel_end_time - kernel_start_time);

    // Verify that the two arrays are equal.
    for (size_t i = 0; i < array_size; i++) {
      if (sum_parallel[i] != sum_sequential[i]) {
        std::cout << "Vector add failed on device.\n";
        return -1;
      }
    }

    int indices[]{0, 1, 2, (static_cast<int>(array_size) - 1)};
    constexpr size_t indices_size = sizeof(indices) / sizeof(int);

    // Print out the result of vector add.
    for (int i = 0; i < indices_size; i++) {
      int j = indices[i];
      if (i == indices_size - 1) std::cout << "...\n";
      std::cout << "[" << j << "]: " << j << " + " << j << " = "
                << sum_sequential[j] << "\n";
    }

    free(a, q);
    free(b, q);
    free(sum_sequential, q);
    free(sum_parallel, q);

    std::cout << "Kernel execution time for " << ITERATIONS
              << " iterations: " << kernel_duration.count()
              << " milliseconds\n";
  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      total_end_time - total_start_time);
  std::cout << "Total execution time: " << total_duration.count()
            << " milliseconds\n";

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}