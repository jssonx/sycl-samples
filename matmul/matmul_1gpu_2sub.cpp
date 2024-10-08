#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sycl/sycl.hpp>

constexpr int m_size = 2200 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;
constexpr int ITERATIONS = 10;
constexpr int VERIFICATION_SAMPLES =
    2000;  // Number of random samples to verify

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

void matmul(sycl::queue& q, float (*a)[N], float (*b)[P], float (*c)[P]);
int verifyResultSingle(float (*c_back)[P], bool full_verify = false);
int verifyResult(std::vector<float (*)[P]>& c_matrices,
                 bool full_verify = false);
void initializeMatrixA(sycl::queue& q, float (*a)[N]);
void initializeMatrixB(sycl::queue& q, float (*b)[P]);

int main(int argc, char* argv[]) {
  std::vector<float(*)[N]> a_matrices(2);
  std::vector<float(*)[P]> b_matrices(2);
  std::vector<float(*)[P]> c_matrices(2);
  std::vector<sycl::queue> queues;
  std::vector<sycl::device> sub_devices;

  bool full_verify = false;
  if (argc > 1 && std::strcmp(argv[1], "--full-verify") == 0) {
    full_verify = true;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    auto l0_selector = [](const sycl::device& dev) {
      return dev.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero;
    };

    sycl::device root_device(l0_selector);
    std::cout << "Main device: "
              << root_device.get_info<sycl::info::device::name>() << "\n";

    // Create sub-devices
    sub_devices = root_device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);

    if (sub_devices.size() < 2) {
      throw std::runtime_error("Not enough sub-devices available");
    }

    // Create queues for each sub-device
    for (int i = 0; i < 2; ++i) {
      queues.emplace_back(sub_devices[i], exception_handler);
      std::cout << "Sub-device " << i << ": "
                << sub_devices[i].get_info<sycl::info::device::name>() << "\n";
    }

    // Allocate USM memory for each sub-device
    for (int i = 0; i < 2; ++i) {
      a_matrices[i] = static_cast<float(*)[N]>(
          sycl::malloc_shared(M * N * sizeof(float), queues[i]));
      b_matrices[i] = static_cast<float(*)[P]>(
          sycl::malloc_shared(N * P * sizeof(float), queues[i]));
      c_matrices[i] = static_cast<float(*)[P]>(
          sycl::malloc_shared(M * P * sizeof(float), queues[i]));

      if (!a_matrices[i] || !b_matrices[i] || !c_matrices[i]) {
        throw std::runtime_error("USM allocation failed for sub-device " +
                                 std::to_string(i));
      }

      initializeMatrixA(queues[i], a_matrices[i]);
      initializeMatrixB(queues[i], b_matrices[i]);
    }

    std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << ","
              << N << ") * b(" << N << "," << P << ")\n";

    for (int iter = 0; iter < ITERATIONS; ++iter) {
      std::cout << "Iteration " << iter + 1 << " of " << ITERATIONS
                << std::endl;

      for (int i = 0; i < 2; ++i) {
        std::cout << "Executing on sub-device " << i << ": " << "\n";
        matmul(queues[i], a_matrices[i], b_matrices[i], c_matrices[i]);
      }

      for (auto& q : queues) {
        q.wait();
      }
    }
  } catch (sycl::exception const& e) {
    std::cout << "An exception is caught while multiplying matrices: "
              << e.what() << "\n";
    // Cleanup
    for (int i = 0; i < 2; ++i) {
      sycl::free(a_matrices[i], queues[i]);
      sycl::free(b_matrices[i], queues[i]);
      sycl::free(c_matrices[i], queues[i]);
    }
    return -1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  std::cout << "Matrix multiplication time: " << duration.count() / 1000000.0
            << " seconds" << std::endl;

  auto verify_start = std::chrono::high_resolution_clock::now();
  int result = verifyResult(c_matrices, full_verify);
  auto verify_end = std::chrono::high_resolution_clock::now();
  auto verify_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      verify_end - verify_start);

  std::cout << "Verification time: " << verify_duration.count() / 1000000.0
            << " seconds" << std::endl;

  // Free USM memory
  for (int i = 0; i < 2; ++i) {
    sycl::free(a_matrices[i], queues[i]);
    sycl::free(b_matrices[i], queues[i]);
    sycl::free(c_matrices[i], queues[i]);
  }

  std::cout << "Total execution time: "
            << (duration + verify_duration).count() / 1000000.0 << " seconds"
            << std::endl;

  return result;
}

void matmul(sycl::queue& q, float (*a)[N], float (*b)[P], float (*c)[P]) {
  q.parallel_for(sycl::range(M, P), [=](sycl::id<2> index) {
     int row = index[0];
     int col = index[1];
     float sum = 0.0f;

     for (int i = 0; i < N; i++) {
       sum += a[row][i] * b[i][col];
     }

     c[row][col] = sum;
   }).wait();
}

bool valueSame(float a, float b) {
  return std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b)) < 1e-4;
}

int verifyResultSingle(float (*c_back)[P], bool full_verify) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis_m(0, M - 1);
  std::uniform_int_distribution<> dis_p(0, P - 1);

  int verified_count = full_verify ? M * P : VERIFICATION_SAMPLES;
  int mismatch_count = 0;

  for (int count = 0; count < verified_count; ++count) {
    int i = full_verify ? count / P : dis_m(gen);
    int j = full_verify ? count % P : dis_p(gen);

    float expected = 0.0f;
    for (int k = 0; k < N; k++) {
      expected += 1.0f * (k + 1.0f);  // a[i][k] * b[k][j]
    }

    if (!valueSame(c_back[i][j], expected)) {
      if (mismatch_count < 5) {
        std::cout << "Mismatch at [" << i << "][" << j << "]: " << "Expected "
                  << expected << ", Got " << c_back[i][j] << "\n";
      }
      mismatch_count++;
    }
  }

  if (mismatch_count == 0) {
    std::cout << "Success - All verified elements are correct!\n";
    return 0;
  } else {
    float mismatch_rate =
        static_cast<float>(mismatch_count) / verified_count * 100;
    std::cout << "Fail - Mismatch rate: " << mismatch_rate << "%\n";
    return -1;
  }
}

int verifyResult(std::vector<float (*)[P]>& c_matrices, bool full_verify) {
  int result = 0;
  for (int i = 0; i < 2; ++i) {
    std::cout << "Verifying result from sub-device " << i << ":\n";
    result |= verifyResultSingle(c_matrices[i], full_verify);
  }
  return result;
}

void initializeMatrixA(sycl::queue& q, float (*a)[N]) {
  q.parallel_for(sycl::range(M, N), [=](sycl::id<2> index) {
     a[index[0]][index[1]] = 1.0f;
   }).wait();
}

void initializeMatrixB(sycl::queue& q, float (*b)[P]) {
  q.parallel_for(sycl::range(N, P), [=](sycl::id<2> index) {
     b[index[0]][index[1]] = index[0] + 1.0f;
   }).wait();
}