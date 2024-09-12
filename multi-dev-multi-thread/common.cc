#include "common.h"
#include <sys/syscall.h>
#include <unistd.h>

cl::sycl::queue createQueue(const cl::sycl::device& device) {
    cl::sycl::queue queue(device, cl::sycl::property::queue::enable_profiling{});

    std::cout << "Created queue on device: "
              << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";
    std::cout << "Max compute units: " << device.get_info<cl::sycl::info::device::max_compute_units>() << "\n";
    std::cout << "Max work-group size: " << device.get_info<cl::sycl::info::device::max_work_group_size>() << "\n";

    return queue;
}

std::vector<cl::sycl::device> initgpu()
{
    try
    {
        // Get all devices of type GPU
        auto platforms = cl::sycl::platform::get_platforms();
        std::vector<cl::sycl::device> gpus;
        std::vector<cl::sycl::device> all_sub_devices;

        for (const auto &platform : platforms)
        {
            auto devices = platform.get_devices(cl::sycl::info::device_type::gpu);
            gpus.insert(gpus.end(), devices.begin(), devices.end());
        }

        if (gpus.empty())
        {
            std::cerr << "No GPU devices found.\n";
            std::terminate();
        }

        // Create sub-devices for each GPU
        for (const auto &gpu : gpus)
        {
            try
            {
                // Try to create sub-devices based on compute units
                auto sub_devices = gpu.create_sub_devices<
                    cl::sycl::info::partition_property::partition_by_affinity_domain>(
                    cl::sycl::info::partition_affinity_domain::next_partitionable);
                
                all_sub_devices.insert(all_sub_devices.end(), sub_devices.begin(), sub_devices.end());
                
                std::cout << "Created " << sub_devices.size() << " sub-devices for GPU: " 
                          << gpu.get_info<cl::sycl::info::device::name>() << "\n";
            }
            catch (cl::sycl::exception &e)
            {
                std::cout << "Failed to create sub-devices for GPU " 
                          << gpu.get_info<cl::sycl::info::device::name>() 
                          << ": " << e.what() << "\n";
                std::cout << "Using the main device as a single sub-device.\n";
                all_sub_devices.push_back(gpu);
            }
        }

        // return gpus;
        return all_sub_devices;
    }
    catch (cl::sycl::exception const &e)
    {
        std::cout << "An exception is caught: " << e.what() << "\n";
        std::terminate();
    }
}

void vecadd_kernel(cl::sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int iteration, const std::string &func_name)
{
    cl::sycl::buffer<int, 1> buffer_a(a.data(), cl::sycl::range<1>(N));
    cl::sycl::buffer<int, 1> buffer_b(b.data(), cl::sycl::range<1>(N));
    cl::sycl::buffer<int, 1> buffer_c(c.data(), cl::sycl::range<1>(N));
    
    pid_t tid = syscall(SYS_gettid);

    std::cout << "Thread " << tid << ", iteration " << iteration << ", " << func_name <<" started.\n";

    cl::sycl::event event = queue.submit([&](cl::sycl::handler &cgh)
                                         {
        // use accessor to access the data in the buffers
        cl::sycl::accessor acc_a(buffer_a, cgh, cl::sycl::read_only);
        cl::sycl::accessor acc_b(buffer_b, cgh, cl::sycl::read_only);
        cl::sycl::accessor acc_c(buffer_c, cgh, cl::sycl::write_only);

        cgh.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::id<1> idx) {
            for (int kk = 0; kk < 10000; kk++) {
                acc_c[idx] = acc_c[idx] + acc_a[N - kk] / double(10000) + acc_b[kk] / double(10000);
            }
        }); });

    event.wait();

    auto start = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    double duration = (end - start) / 1e3;

    std::cout << "Thread " << tid << ", iteration " << iteration << ", " << func_name <<" executed in " << duration << " us. " << "Kernel start: " << start / 1e3 << " us, end: " << end / 1e3 << "\n";
}