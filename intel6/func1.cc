#include "common.h"

void kernel_submission_1(cl::sycl::queue queue, int thread_id, size_t X) {
    std::vector<int> a(X, thread_id);
    std::vector<int> b(X, thread_id);
    std::vector<int> c(X, 0);

    for (size_t i = 0; i < LOOP_COUNT; ++i) {
        vecadd_kernel(queue, a, b, c, X, thread_id, i);
    }
}