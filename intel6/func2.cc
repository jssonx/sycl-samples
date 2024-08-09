#include "common.h"

void kernel_submission_2(cl::sycl::queue queue, int thread_id) {
    std::vector<int> a(N_SINGLE, thread_id);
    std::vector<int> b(N_SINGLE, thread_id);
    std::vector<int> c(N_SINGLE, 0);

    for (size_t i = 0; i < 1; ++i) {
        vecadd_kernel(queue, a, b, c, N_SINGLE, thread_id, i);
    }
}