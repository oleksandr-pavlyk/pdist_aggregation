#include <CL/sycl.hpp>
#include <limits>
#include <random>
#include <chrono>
#include "heap.hpp"
#include "parallel_knn.hpp"

static inline double elapsed(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-06;
}

int main(void)
{
    sycl::queue q;
    size_t n_train = 1024 * 1024;
    size_t dim = 5;
    size_t n_test = 273;

    double *X_train_host = new double[n_train * dim];
    double *X_test_host = new double[n_train * dim];

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for (size_t i = 0; i < n_train * dim; ++i)
        {
            X_train_host[i] = dis(gen);
        }
        for (size_t i = 0; i < n_test * dim; ++i)
        {
            X_test_host[i] = dis(gen);
        }
    }
    size_t k = 5;
    double *knn_dists = new double[n_test * k];
    long long *knn_ind = new long long[n_test * k];

#if 0
    sycl::vector_class<sycl::kernel_id> kernel_ids_vec = {
	sycl::get_kernel_id<kern_topk>(),
	sycl::get_kernel_id<kern_pwdist>(),
	sycl::get_kernel_id<kern_init_max_heap>(),
	sycl::get_kernel_id<kern_init_aggregation_heap>()
    };

    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), kernel_ids_vec);
#endif

    auto t0 = std::chrono::steady_clock::now();
    parallel_knn_search<double, long long>(q, k, dim, X_train_host, n_train, X_test_host, n_test, knn_ind, knn_dists, {});
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;

    t0 = std::chrono::steady_clock::now();
    parallel_knn_search<double, long long>(q, k, dim, X_train_host, n_train, X_test_host, n_test, knn_ind, knn_dists, {});
    q.wait();
    t1 = std::chrono::steady_clock::now();
    std::cout << "Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;

    std::cout
        << std::endl;
    for (size_t i_test = 0; i_test < n_test; ++i_test)
    {
        for (size_t j = 0; j < k; ++j)
        {
            std::cout << knn_ind[i_test * k + j] << "[" << knn_dists[i_test * k + j] << "] ";
        }
        std::cout << std::endl;
    }

    delete[] X_train_host;
    delete[] X_test_host;
    delete[] knn_dists;
    delete[] knn_ind;

    return 0;
}
