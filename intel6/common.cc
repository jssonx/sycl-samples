#include "common.h"

cl::sycl::queue initgpu() {
    try {
        cl::sycl::queue queue(cl::sycl::property::queue::enable_profiling{});
        // Print out the device information used for the kernel code.
        std::cout << "Running on device: "
                  << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";
        
        auto device = queue.get_device();
        std::cout << "Max compute units: " << device.get_info<cl::sycl::info::device::max_compute_units>() << "\n";
        std::cout << "Max work-group size: " << device.get_info<cl::sycl::info::device::max_work_group_size>() << "\n";
        std::cout << "Device name: " << device.get_info<cl::sycl::info::device::name>() << "\n";

        return queue;
    } catch (cl::sycl::exception const &e) {
        std::cout << "An exception is caught trying to determine device.\n";
        std::terminate();
    }
}

void vecadd_kernel(cl::sycl::queue &queue, std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, size_t N, int thread_id, int iteration) {
    cl::sycl::buffer<int, 1> buffer_a(a.data(), cl::sycl::range<1>(N));
    cl::sycl::buffer<int, 1> buffer_b(b.data(), cl::sycl::range<1>(N));
    cl::sycl::buffer<int, 1> buffer_c(c.data(), cl::sycl::range<1>(N));

    cl::sycl::event event = queue.submit([&](cl::sycl::handler &cgh) {
        // use accessor to access the data in the buffers
        cl::sycl::accessor acc_a(buffer_a, cgh, cl::sycl::read_only);
        cl::sycl::accessor acc_b(buffer_b, cgh, cl::sycl::read_only);
        cl::sycl::accessor acc_c(buffer_c, cgh, cl::sycl::write_only);

        cgh.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::id<1> idx) {
            for (int kk = 0; kk < 10000; kk++) {
                acc_c[idx] = acc_c[idx] + acc_a[N - kk] / double(10000) + acc_b[kk] / double(10000);
            }
        });
    });

    event.wait();

    auto start = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    double duration = (end - start) / 1e3;

    std::cout << "Thread " << thread_id << ", iteration " << iteration << ", kernel executed in " << duration << " us. " << "Kernel start: " << start / 1e3 << " us, end: " << end / 1e3 << "\n";
}