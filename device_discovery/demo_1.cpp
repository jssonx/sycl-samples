#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

int main() {
  try {
    // 1. Device discovery
    auto platforms = platform::get_platforms();
    auto gpu_platform =
        std::find_if(platforms.begin(), platforms.end(), [](const platform &p) {
          return p.get_info<info::platform::name>().find("Level-Zero") !=
                 std::string::npos;
        });

    if (gpu_platform == platforms.end()) {
      std::cout << "No Level-Zero GPU platform found.\n";
      return 1;
    }

    auto devices = gpu_platform->get_devices();
    if (devices.empty()) {
      std::cout << "No GPU devices found.\n";
      return 1;
    }

    std::cout << "Number of root devices: " << devices.size() << std::endl;

    // 2. Create sub-devices (if supported)
    std::vector<device> all_sub_devices;
    for (auto &dev : devices) {
      std::cout << "Device: " << dev.get_info<info::device::name>()
                << std::endl;
      try {
        auto sub_devices = dev.create_sub_devices<
            info::partition_property::partition_by_affinity_domain>(
            info::partition_affinity_domain::next_partitionable);
        all_sub_devices.insert(all_sub_devices.end(), sub_devices.begin(),
                               sub_devices.end());
        std::cout << "  Number of sub-devices: " << sub_devices.size()
                  << std::endl;
      } catch (exception &e) {
        std::cout << "  Failed to create sub-devices. Error: " << e.what()
                  << std::endl;
        std::cout << "  This device will be treated as a single sub-device."
                  << std::endl;
        all_sub_devices.push_back(dev);
      }
    }

    std::cout
        << "Total number of sub-devices (including non-partitionable devices): "
        << all_sub_devices.size() << std::endl;
  } catch (exception &e) {
    std::cout << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}