#include "common.h"

void kernel_submission(cl::sycl::queue queue, size_t X, const std::string& func_name)
{
    std::vector<int> a(X, 2);
    std::vector<int> b(X, 5);
    std::vector<int> c(X, 0);

    for (size_t i = 0; i < LOOP_COUNT; ++i)
    {
        vecadd_kernel(queue, a, b, c, X, i, func_name);
    }
}