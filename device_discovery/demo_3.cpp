#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

void
vector_add(queue& q, int* a, int* b, int* c, size_t n)
{
  q.parallel_for(range<1>(n), [=](id<1> i) { c[i] = a[i] + b[i]; }).wait();
}

int
main()
{
  constexpr size_t N = 1024 * 1024;

  try
  {
    device gpu_device;
    auto platforms = platform::get_platforms();
    for (auto& platform : platforms)
    {
      auto devices = platform.get_devices(info::device_type::gpu);
      if (!devices.empty())
      {
        gpu_device = devices[0];
        break;
      }
    }

    if (gpu_device.get_info<info::device::name>().empty())
    {
      std::cout << "No GPU device found.\n";
      return 1;
    }

    std::cout << "Selected device: "
              << gpu_device.get_info<info::device::name>() << std::endl;

    std::vector<device> sub_devices;
    try
    {
      sub_devices = gpu_device.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
    }
    catch (exception& e)
    {
      std::cout << "Failed to create sub-devices: " << e.what() << std::endl;
      std::cout << "Using the main device as a single sub-device.\n";
      sub_devices.push_back(gpu_device);
    }

    std::cout << "Number of sub-devices: " << sub_devices.size() << std::endl;

    std::vector<queue> queues;
    for (const auto& sub_dev : sub_devices)
    {
      queues.emplace_back(sub_dev);
    }

    // Allocate USM memory
    queue& main_queue = queues[0];
    int* a = malloc_shared<int>(N, main_queue);
    int* b = malloc_shared<int>(N, main_queue);
    int* c = malloc_shared<int>(N, main_queue);

    // Initialize data
    for (size_t i = 0; i < N; ++i)
    {
      a[i] = 1;
      b[i] = 2;
      c[i] = 0;
    }

    size_t sub_size = N / sub_devices.size();
    for (size_t i = 0; i < sub_devices.size(); ++i)
    {
      size_t offset = i * sub_size;
      size_t end = (i == sub_devices.size() - 1) ? N : (i + 1) * sub_size;
      size_t local_size = end - offset;

      vector_add(queues[i], a + offset, b + offset, c + offset, local_size);
    }

    for (auto& q : queues)
    {
      q.wait();
    }

    bool correct = true;
    for (size_t i = 0; i < N; ++i)
    {
      if (c[i] != 3)
      {
        correct = false;
        break;
      }
    }

    std::cout << "Computation " << (correct ? "succeeded" : "failed")
              << std::endl;

    // Free USM memory
    free(a, main_queue);
    free(b, main_queue);
    free(c, main_queue);
  }
  catch (exception& e)
  {
    std::cout << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}