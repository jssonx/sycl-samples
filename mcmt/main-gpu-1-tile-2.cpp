#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

// Array size for this example.
size_t array_size = 100000000;

// Create an exception handler for asynchronous SYCL exceptions
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

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum".
//************************************
void
VectorAdd(queue &q1,
          queue &q2,
          const int *a,
          const int *b,
          int *sum,
          size_t size)
{
  size_t half_size = size / 2;
  range<1> num_items_half{half_size};

  auto e1 =
      q1.parallel_for(num_items_half, [=](auto i) { sum[i] = a[i] + b[i]; });

  auto e2 = q2.parallel_for(num_items_half,
                            [=](auto i)
                            {
                              size_t offset = half_size;
                              sum[i + offset] = a[i + offset] + b[i + offset];
                            });

  e1.wait();
  e2.wait();
}

//************************************
// Initialize the array from 0 to array_size - 1
//************************************
void
InitializeArray(int *a, size_t size)
{
  for (size_t i = 0; i < size; i++) a[i] = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int
main(int argc, char *argv[])
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // Change array_size if it was passed as argument
  if (argc > 1) array_size = std::stoi(argv[1]);

  try
  {
    // Select GPU device
    device gpu_device;
    auto platforms = platform::get_platforms();
    for (auto &platform : platforms)
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

    // Create sub-devices
    std::vector<device> sub_devices;
    try
    {
      sub_devices = gpu_device.create_sub_devices<
          info::partition_property::partition_by_affinity_domain>(
          info::partition_affinity_domain::next_partitionable);
    }
    catch (exception &e)
    {
      std::cout << "Failed to create sub-devices: " << e.what() << std::endl;
      std::cout << "Using the main device as a single sub-device.\n";
      sub_devices.push_back(gpu_device);
    }

    std::cout << "Number of sub-devices: " << sub_devices.size() << std::endl;

    // Create queues for each sub-device
    queue q1(sub_devices[0], exception_handler);
    queue q2(sub_devices[1], exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << gpu_device.get_info<info::device::name>() << "\n";
    std::cout << "Number of sub-devices used: " << sub_devices.size() << "\n";
    std::cout << "Vector size: " << array_size << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    int *a = malloc_shared<int>(array_size, q1);
    int *b = malloc_shared<int>(array_size, q1);
    int *sum_sequential = malloc_shared<int>(array_size, q1);
    int *sum_parallel = malloc_shared<int>(array_size, q1);

    if ((a == nullptr) || (b == nullptr) || (sum_sequential == nullptr) ||
        (sum_parallel == nullptr))
    {
      if (a != nullptr) free(a, q1);
      if (b != nullptr) free(b, q1);
      if (sum_sequential != nullptr) free(sum_sequential, q1);
      if (sum_parallel != nullptr) free(sum_parallel, q1);

      std::cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Initialize input arrays with values from 0 to array_size - 1
    InitializeArray(a, array_size);
    InitializeArray(b, array_size);

    // Compute the sum of two arrays in sequential for validation.
    for (size_t i = 0; i < array_size; i++) sum_sequential[i] = a[i] + b[i];

    // Vector addition in SYCL using two sub-devices.
    VectorAdd(q1, q2, a, b, sum_parallel, array_size);

    // Verify that the two arrays are equal.
    for (size_t i = 0; i < array_size; i++)
    {
      if (sum_parallel[i] != sum_sequential[i])
      {
        std::cout << "Vector add failed on device.\n";
        return -1;
      }
    }

    int indices[]{0, 1, 2, (static_cast<int>(array_size) - 1)};
    constexpr size_t indices_size = sizeof(indices) / sizeof(int);

    // Print out the result of vector add.
    for (int i = 0; i < indices_size; i++)
    {
      int j = indices[i];
      if (i == indices_size - 1) std::cout << "...\n";
      std::cout << "[" << j << "]: " << j << " + " << j << " = "
                << sum_sequential[j] << "\n";
    }

    free(a, q1);
    free(b, q1);
    free(sum_sequential, q1);
    free(sum_parallel, q1);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  std::cout << "Total execution time: " << duration.count()
            << " milliseconds\n";

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}