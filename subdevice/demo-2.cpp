
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

void
vector_add(queue& q,
           buffer<int>& a_buf,
           buffer<int>& b_buf,
           buffer<int>& c_buf,
           size_t n)
{
  q.submit(
      [&](handler& h)
      {
        auto a = a_buf.get_access<access::mode::read>(h);
        auto b = b_buf.get_access<access::mode::read>(h);
        auto c = c_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(n), [=](id<1> i) { c[i] = a[i] + b[i]; });
      });
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

    std::vector<int> a(N, 1), b(N, 2), c(N, 0);

    size_t sub_size = N / sub_devices.size();
    for (size_t i = 0; i < sub_devices.size(); ++i)
    {
      size_t offset = i * sub_size;
      size_t end = (i == sub_devices.size() - 1) ? N : (i + 1) * sub_size;
      size_t local_size = end - offset;

      buffer<int> a_buf(a.data() + offset, range<1>(local_size));
      buffer<int> b_buf(b.data() + offset, range<1>(local_size));
      buffer<int> c_buf(c.data() + offset, range<1>(local_size));

      vector_add(queues[i], a_buf, b_buf, c_buf, local_size);
    }

    for (auto& q : queues)
    {
      q.wait_and_throw();
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
  }
  catch (exception& e)
  {
    std::cout << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}