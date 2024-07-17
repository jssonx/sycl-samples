#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

size_t array_size = 100000000;

static auto exception_handler = [](sycl::exception_list e_list)
{
  for (std::exception_ptr const &e : e_list)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e)
    {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

void
VectorAdd(queue &q, const int *a, const int *b, int *sum, size_t size)
{
  range<1> num_items{size};
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
  e.wait();
}

int
main(int argc, char *argv[])
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (argc > 1) array_size = std::stoi(argv[1]);

  try
  {
    // Get all GPU devices
    auto platforms = platform::get_platforms();
    std::vector<device> gpu_devices;
    for (auto &platform : platforms)
    {
      auto devices = platform.get_devices(info::device_type::gpu);
      gpu_devices.insert(gpu_devices.end(), devices.begin(), devices.end());
    }

    if (gpu_devices.size() < 2)
    {
      std::cout << "Not enough GPU devices available. At least 2 GPUs are "
                   "required.\n";
      return -1;
    }

    // Explicitly use device 0 and device 1
    std::vector<queue> queues;
    queues.emplace_back(gpu_devices[0], exception_handler);
    queues.emplace_back(gpu_devices[1], exception_handler);

    std::cout << "Using device 0: "
              << gpu_devices[0].get_info<info::device::name>() << "\n";
    std::cout << "Using device 1: "
              << gpu_devices[1].get_info<info::device::name>() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Allocate memory for each device
    std::vector<int *> a_list(2);
    std::vector<int *> b_list(2);
    std::vector<int *> sum_parallel_list(2);
    int *sum_sequential = malloc_shared<int>(array_size, queues[0]);

    size_t sub_size = array_size / 2;
    for (int i = 0; i < 2; ++i)
    {
      size_t local_size = (i == 1) ? (array_size - sub_size) : sub_size;
      a_list[i] = malloc_shared<int>(local_size, queues[i]);
      b_list[i] = malloc_shared<int>(local_size, queues[i]);
      sum_parallel_list[i] = malloc_shared<int>(local_size, queues[i]);
    }

    // Initialize arrays
    for (int i = 0; i < 2; ++i)
    {
      size_t offset = i * sub_size;
      size_t local_size = (i == 1) ? (array_size - sub_size) : sub_size;
      for (size_t j = 0; j < local_size; ++j)
      {
        a_list[i][j] = offset + j;
        b_list[i][j] = offset + j;
      }
    }

    // Sequential computation for verification
    for (size_t i = 0; i < array_size; i++)
    {
      sum_sequential[i] = i + i;
    }

    // Parallel computation on two devices
    for (int i = 0; i < 2; ++i)
    {
      size_t local_size = (i == 1) ? (array_size - sub_size) : sub_size;
      VectorAdd(queues[i], a_list[i], b_list[i], sum_parallel_list[i],
                local_size);
    }

    // Wait for all queues to finish
    for (auto &q : queues)
    {
      q.wait_and_throw();
    }

    // Verify the results
    bool correct = true;
    for (int i = 0; i < 2; ++i)
    {
      size_t offset = i * sub_size;
      size_t local_size = (i == 1) ? (array_size - sub_size) : sub_size;
      for (size_t j = 0; j < local_size; ++j)
      {
        if (sum_parallel_list[i][j] != sum_sequential[offset + j])
        {
          correct = false;
          std::cout << "Mismatch at device " << i << ", local index " << j
                    << ", global index " << (offset + j)
                    << ". Expected: " << sum_sequential[offset + j]
                    << ", Got: " << sum_parallel_list[i][j] << "\n";
          // Do not exit immediately, continue to check to show more error
          // information
        }
      }
    }

    if (correct)
    {
      std::cout << "Vector addition is correct on both devices.\n";
    }
    else
    {
      std::cout << "Vector addition failed.\n";
    }

    // Clean up
    for (int i = 0; i < 2; ++i)
    {
      free(a_list[i], queues[i]);
      free(b_list[i], queues[i]);
      free(sum_parallel_list[i], queues[i]);
    }
    free(sum_sequential, queues[0]);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught: " << e.what() << "\n";
    return -1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Total execution time: " << duration.count()
            << " milliseconds\n";

  std::cout << "Vector add completed on two devices.\n";
  return 0;
}