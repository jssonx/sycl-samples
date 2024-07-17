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
  for (std::exception_ptr const& e : e_list)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (std::exception const& e)
    {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

void
VectorAdd(queue& q, const int* a, const int* b, int* sum, size_t size)
{
  range<1> num_items{size};
  auto e = q.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
  e.wait();
}

int
main(int argc, char* argv[])
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (argc > 1) array_size = std::stoi(argv[1]);

  try
  {
    // Get all GPU devices
    auto platforms = platform::get_platforms();
    std::vector<device> gpu_devices;
    for (auto& platform : platforms)
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

    // Use device 0 and device 1
    std::vector<device> main_devices = {gpu_devices[0], gpu_devices[1]};
    std::vector<std::vector<device>> all_sub_devices;

    for (int i = 0; i < 2; ++i)
    {
      std::cout << "Main device " << i << ": "
                << main_devices[i].get_info<info::device::name>() << "\n";
      std::vector<device> sub_devices;
      try
      {
        sub_devices =
            main_devices[i]
                .create_sub_devices<
                    info::partition_property::partition_by_affinity_domain>(
                    info::partition_affinity_domain::next_partitionable);
        std::cout << "  Number of sub-devices: " << sub_devices.size() << "\n";
      }
      catch (exception& e)
      {
        std::cout << "  Failed to create sub-devices: " << e.what() << "\n";
        std::cout << "  Using the main device as a single sub-device.\n";
        sub_devices.push_back(main_devices[i]);
      }
      all_sub_devices.push_back(sub_devices);
    }

    std::cout << "Vector size: " << array_size << "\n";

    // Create queues for all sub-devices
    std::vector<std::vector<queue>> all_queues;
    for (const auto& sub_devices : all_sub_devices)
    {
      std::vector<queue> device_queues;
      for (const auto& sub_dev : sub_devices)
      {
        device_queues.emplace_back(sub_dev, exception_handler);
      }
      all_queues.push_back(device_queues);
    }

    // Allocate memory for each sub-device
    std::vector<std::vector<int*>> a_list(2), b_list(2), sum_parallel_list(2);
    int* sum_sequential = malloc_shared<int>(array_size, all_queues[0][0]);

    size_t main_sub_size = array_size / 2;
    for (int i = 0; i < 2; ++i)
    {
      size_t sub_device_count = all_sub_devices[i].size();
      size_t sub_size = main_sub_size / sub_device_count;
      for (size_t j = 0; j < sub_device_count; ++j)
      {
        size_t local_size = (j == sub_device_count - 1)
                                ? (main_sub_size - j * sub_size)
                                : sub_size;
        a_list[i].push_back(malloc_shared<int>(local_size, all_queues[i][j]));
        b_list[i].push_back(malloc_shared<int>(local_size, all_queues[i][j]));
        sum_parallel_list[i].push_back(
            malloc_shared<int>(local_size, all_queues[i][j]));
      }
    }

    // Initialize arrays
    for (int i = 0; i < 2; ++i)
    {
      size_t main_offset = i * main_sub_size;
      size_t sub_device_count = all_sub_devices[i].size();
      size_t sub_size = main_sub_size / sub_device_count;
      for (size_t j = 0; j < sub_device_count; ++j)
      {
        size_t local_offset = j * sub_size;
        size_t local_size = (j == sub_device_count - 1)
                                ? (main_sub_size - j * sub_size)
                                : sub_size;
        for (size_t k = 0; k < local_size; ++k)
        {
          size_t global_index = main_offset + local_offset + k;
          a_list[i][j][k] = global_index;
          b_list[i][j][k] = global_index;
        }
      }
    }

    // Sequential computation for verification
    for (size_t i = 0; i < array_size; i++)
    {
      sum_sequential[i] = i + i;
    }

    // Compute in parallel on all sub-devices
    for (int i = 0; i < 2; ++i)
    {
      size_t sub_device_count = all_sub_devices[i].size();
      size_t sub_size = main_sub_size / sub_device_count;
      for (size_t j = 0; j < sub_device_count; ++j)
      {
        size_t local_size = (j == sub_device_count - 1)
                                ? (main_sub_size - j * sub_size)
                                : sub_size;
        VectorAdd(all_queues[i][j], a_list[i][j], b_list[i][j],
                  sum_parallel_list[i][j], local_size);
      }
    }

    // Wait for all queues to complete
    for (auto& device_queues : all_queues)
    {
      for (auto& q : device_queues)
      {
        q.wait_and_throw();
      }
    }

    // Verification
    bool correct = true;
    for (int i = 0; i < 2; ++i)
    {
      size_t main_offset = i * main_sub_size;
      size_t sub_device_count = all_sub_devices[i].size();
      size_t sub_size = main_sub_size / sub_device_count;
      for (size_t j = 0; j < sub_device_count; ++j)
      {
        size_t local_offset = j * sub_size;
        size_t local_size = (j == sub_device_count - 1)
                                ? (main_sub_size - j * sub_size)
                                : sub_size;
        for (size_t k = 0; k < local_size; ++k)
        {
          size_t global_index = main_offset + local_offset + k;
          if (sum_parallel_list[i][j][k] != sum_sequential[global_index])
          {
            correct = false;
            std::cout << "Mismatch at device " << i << ", sub-device " << j
                      << ", local index " << k << ", global index "
                      << global_index
                      << ". Expected: " << sum_sequential[global_index]
                      << ", Got: " << sum_parallel_list[i][j][k] << "\n";
          }
        }
      }
    }

    if (correct)
    {
      std::cout
          << "Vector addition is correct on all devices and sub-devices.\n";
    }
    else
    {
      std::cout << "Vector addition failed.\n";
    }

    // Clean up
    for (int i = 0; i < 2; ++i)
    {
      for (size_t j = 0; j < all_sub_devices[i].size(); ++j)
      {
        free(a_list[i][j], all_queues[i][j]);
        free(b_list[i][j], all_queues[i][j]);
        free(sum_parallel_list[i][j], all_queues[i][j]);
      }
    }
    free(sum_sequential, all_queues[0][0]);
  }
  catch (exception const& e)
  {
    std::cout << "An exception is caught: " << e.what() << "\n";
    return -1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Total execution time: " << duration.count()
            << " milliseconds\n";

  std::cout << "Vector add completed on two devices with sub-devices.\n";
  return 0;
}