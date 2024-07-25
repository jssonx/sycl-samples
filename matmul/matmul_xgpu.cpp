#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sycl/sycl.hpp>

constexpr int m_size = 7200 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;
constexpr int ITERATIONS = 25;
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
int verifyResult(float (*c_back)[P], bool full_verify = false);
void initializeMatrixA(sycl::queue& q, float (*a)[N]);
void initializeMatrixB(sycl::queue& q, float (*b)[P]);

int main(int argc, char* argv[]) {
  int num_gpu = 6;
  bool full_verify = false;
  
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--full-verify") == 0) {
      full_verify = true;
    } else {
      num_gpu = std::stoi(argv[i]);
    }
  }

  std::vector<float(*)[N]> a_matrices(num_gpu);
  std::vector<float(*)[P]> b_matrices(num_gpu);
  std::vector<float(*)[P]> c_matrices(num_gpu);
  std::vector<sycl::queue> queues;
  std::vector<sycl::context> contexts;

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpu_devices;
    for (auto& platform : platforms) {
      if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        continue;  // Skip non-Level Zero backends
      }
      auto devices = platform.get_devices(sycl::info::device_type::gpu);
      gpu_devices.insert(gpu_devices.end(), devices.begin(), devices.end());
    }

    std::cout << "Number of GPU devices: " << gpu_devices.size() << "\n";

    if (gpu_devices.size() < num_gpu) {
      std::cout << "Not enough GPU devices available.\n";
      return -1;
    }

    // Create queues for each GPU device
    for (int i = 0; i < num_gpu; ++i) {
      contexts.emplace_back(gpu_devices[i]);
      queues.emplace_back(contexts[i], gpu_devices[i], exception_handler);
      std::cout << "Using device " << i << ": "
                << gpu_devices[i].get_info<sycl::info::device::name>() << "\n";
    }

    for (int i = 0; i < num_gpu; ++i) {
      a_matrices[i] = static_cast<float(*)[N]>(
          sycl::malloc_shared(M * N * sizeof(float), queues[i]));
      b_matrices[i] = static_cast<float(*)[P]>(
          sycl::malloc_shared(N * P * sizeof(float), queues[i]));
      c_matrices[i] = static_cast<float(*)[P]>(
          sycl::malloc_shared(M * P * sizeof(float), queues[i]));

      if (!a_matrices[i] || !b_matrices[i] || !c_matrices[i]) {
        throw std::runtime_error("USM allocation failed for device " +
                                 std::to_string(i));
      }

      initializeMatrixA(queues[i], a_matrices[i]);
      initializeMatrixB(queues[i], b_matrices[i]);
    }

    std::cout << "Problem size: c(" << M << "x" << P << ") = a(" << M << "x"
              << N << ") * b(" << N << "x" << P << ")\n";

    for (int iter = 0; iter < ITERATIONS; ++iter) {
      std::cout << "Iteration " << iter + 1 << " of " << ITERATIONS
                << std::endl;

      for (int i = 0; i < num_gpu; ++i) {
        matmul(queues[i], a_matrices[i], b_matrices[i], c_matrices[i]);
      }

      // Wait for all queues to finish
      for (auto& q : queues) {
        q.wait_and_throw();
      }
    }

  } catch (sycl::exception const& e) {
    std::cout << "An exception is caught while multiplying matrices: "
              << e.what() << "\n";

    // Cleanup
    for (int i = 0; i < num_gpu; ++i) {
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
  int result = 0;
  for (int i = 0; i < num_gpu; ++i) {
    result = verifyResult(c_matrices[i], full_verify);
    if (result != 0) {
      std::cout << "Verification failed on device " << i << "\n";
      return result;
    }
  }
  auto verify_end = std::chrono::high_resolution_clock::now();
  auto verify_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      verify_end - verify_start);

  std::cout << "Verification time: " << verify_duration.count() / 1000000.0
            << " seconds" << std::endl;

  // Free USM memory
  for (int i = 0; i < num_gpu; ++i) {
    sycl::free(a_matrices[i], queues[i]);
    sycl::free(b_matrices[i], queues[i]);
    sycl::free(c_matrices[i], queues[i]);
  }

  std::cout << "Total execution time: "
            << (duration + verify_duration).count() / 1000000.0 << " seconds"
            << std::endl;
  
  return 0;
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
  // return std::fabs(a - b) < std::numeric_limits<float>::epsilon() * 100;
  return std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b)) < 1e-4;
}

int verifyResult(float (*c_back)[P], bool full_verify) {
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