#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <chrono>
#include <random>
#include <cstring>

constexpr int m_size = 2400 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;
constexpr int ITERATIONS = 10;
constexpr int VERIFICATION_SAMPLES = 2000; // Number of random samples to verify

// Function to perform matrix multiplication
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

int VerifyResult(float (*c_back)[P], bool full_verify = false);

int main(int argc, char* argv[]) {
  float (*a)[N] = nullptr;
  float (*b)[P] = nullptr;
  float (*c_back)[P] = nullptr;
  sycl::queue q;

  bool full_verify = false;
  if (argc > 1 && std::strcmp(argv[1], "--full-verify") == 0) {
    full_verify = true;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    q = sycl::queue(sycl::default_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Allocate USM memory
    a = static_cast<float(*)[N]>(sycl::malloc_shared(M * N * sizeof(float), q));
    b = static_cast<float(*)[P]>(sycl::malloc_shared(N * P * sizeof(float), q));
    c_back = static_cast<float(*)[P]>(sycl::malloc_shared(M * P * sizeof(float), q));

    if (!a || !b || !c_back) {
      throw std::runtime_error("USM allocation failed");
    }

    std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
              << ") * b(" << N << "," << P << ")\n";

    // Initialize matrices
    q.parallel_for(sycl::range(M, N), [=](sycl::id<2> index) {
      a[index[0]][index[1]] = 1.0f;
    }).wait();

    q.parallel_for(sycl::range(N, P), [=](sycl::id<2> index) {
      b[index[0]][index[1]] = index[0] + 1.0f;
    }).wait();

    for (int iter = 0; iter < ITERATIONS; ++iter) {
      std::cout << "Iteration " << iter + 1 << " of " << ITERATIONS << std::endl;

      // Call the matmul function
      matmul(q, a, b, c_back);
    }

  } catch (sycl::exception const &e) {
    std::cout << "An exception is caught while multiplying matrices: " << e.what() << "\n";
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c_back, q);
    return -1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  std::cout << "Matrix multiplication time: " << duration.count() / 1000000.0 << " seconds" << std::endl;

  auto verify_start = std::chrono::high_resolution_clock::now();
  int result = VerifyResult(c_back, full_verify);
  auto verify_end = std::chrono::high_resolution_clock::now();
  auto verify_duration = std::chrono::duration_cast<std::chrono::microseconds>(verify_end - verify_start);

  std::cout << "Verification time: " << verify_duration.count() / 1000000.0 << " seconds" << std::endl;

  // Free USM memory
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c_back, q);

  std::cout << "Total execution time: " << (duration + verify_duration).count() / 1000000.0 << " seconds" << std::endl;

  return result;
}

bool ValueSame(float a, float b) {
  return std::fabs(a - b) < std::numeric_limits<float>::epsilon() * 100;
}

int VerifyResult(float (*c_back)[P], bool full_verify) {
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

    if (!ValueSame(c_back[i][j], expected)) {
      if (mismatch_count < 5) {
        std::cout << "Mismatch at [" << i << "][" << j << "]: "
                  << "Expected " << expected << ", Got " << c_back[i][j] << "\n";
      }
      mismatch_count++;
    }
  }

  if (mismatch_count == 0) {
    std::cout << "Success - All verified elements are correct!\n";
    return 0;
  } else {
    float mismatch_rate = static_cast<float>(mismatch_count) / verified_count * 100;
    std::cout << "Fail - Mismatch rate: " << mismatch_rate << "%\n";
    return -1;
  }
}