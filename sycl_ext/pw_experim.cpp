#include <CL/sycl.hpp>
#include <chrono>
#include <limits>
#include <random>
#include <cmath>
#include "oneapi/mkl.hpp"

template <typename T>
inline T ceiling_quotient(T n, T m)
{
    return ((n + m - 1) / m);
}

template <typename floatT>
sycl::event pw_dsq_usm(
    sycl::queue &exec_q,
    size_t dim,
    floatT *X_train_ptr,
    size_t n_train,
    floatT *X_test_ptr,
    size_t n_test,
    floatT *dm_ptr,
    size_t ld_dm,
    const sycl::vector_class<sycl::event> &depends = {})
{
    constexpr int dim_chunk = 8;
    static_assert(0 == (dim_chunk & (dim_chunk - 1)));

    assert(sycl::get_pointer_type(X_train_ptr, exec_q.get_context()) != sycl::usm::alloc::unknown);
    assert(sycl::get_pointer_type(X_test_ptr, exec_q.get_context()) != sycl::usm::alloc::unknown);

    auto fill_ev = (dim > dim_chunk) ? exec_q.memset(dm_ptr, 0, n_train * n_test * sizeof(double))
                                     : sycl::event();

    auto res_ev =
        exec_q.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(depends);
                cgh.depends_on(fill_ev);
                sycl::accessor<floatT, 1, sycl::access::mode::read_write, sycl::access::target::local>
                    scratch(dim_chunk, cgh);
                auto ndRange = sycl::nd_range<3>(
                    sycl::range<3>(n_train, n_test, dim_chunk * ceiling_quotient(dim, static_cast<size_t>(dim_chunk))),
                    sycl::range<3>(1, 1, dim_chunk));
                auto kern_func = [=](sycl::nd_item<3> it)
                {
                    auto i_train = it.get_global_id(0);
                    auto i_test = it.get_global_id(1);
                    auto gid = it.get_global_id(2);
                    auto lid = it.get_local_id(2);

                    if (gid < dim)
                    {
                        const floatT diff = (X_train_ptr[i_train * dim + gid] -
                                             X_test_ptr[i_test * dim + gid]);
                        scratch[lid] = diff * diff;
                    }
                    else
                    {
                        scratch[lid] = floatT(0);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (dim_chunk == 8)
                    {
                        int i = dim_chunk >> 1; // 4
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[lid] += scratch[lid + i];
                        it.barrier(sycl::access::fence_space::local_space);
                        i = dim_chunk >> 2; // 2
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[lid] += scratch[lid + i];
                        it.barrier(sycl::access::fence_space::local_space);
                        i = dim_chunk >> 3; // 1
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[lid] += scratch[lid + i];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    else
                    {
                        for (int i = dim_chunk / 2; i > 0; i >>= 1)
                        {
                            if (lid < static_cast<decltype(lid)>(i))
                                scratch[lid] += scratch[lid + i];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                    }
                    if (lid == 0)
                    {
                        if (dim > dim_chunk)
                        {
                            auto dij = sycl::ONEAPI::atomic_ref<
                                floatT,
                                sycl::ONEAPI::memory_order::relaxed,
                                sycl::ONEAPI::memory_scope::device,
                                sycl::access::address_space::global_space>(dm_ptr[i_test * ld_dm + i_train]);
                            dij.fetch_add(scratch[0]);
                        }
                        else
                        {
                            dm_ptr[i_test * ld_dm + i_train] = scratch[0];
                        }
                    }
                };
                cgh.parallel_for(ndRange, kern_func);
            });

    return res_ev;
}

template <typename floatT>
sycl::event pw_dsq_host(
    sycl::queue &exec_q,
    size_t dim,
    floatT *X_train_host_ptr,
    size_t n_train,
    floatT *X_test_host_ptr,
    size_t n_test,
    floatT *dm_host_ptr,
    size_t ld_dm,
    const sycl::vector_class<sycl::event> &depends = {})
{
    constexpr int dim_chunk = 2;
    constexpr int n_train_chunk = 16;
    constexpr int n_test_chunk = 8;
    static_assert(0 == (dim_chunk & (dim_chunk - 1)));
    static_assert(dim_chunk == 2 || dim_chunk == 4 || dim_chunk == 8);

    sycl::buffer<floatT, 1> X_train_buf(X_train_host_ptr, sycl::range<1>(n_train * dim));
    sycl::buffer<floatT, 1> X_test_buf(X_test_host_ptr, sycl::range<1>(n_test * dim));

    if (dim > dim_chunk)
    {
        memset(dm_host_ptr, 0, n_train * n_test * sizeof(double));
    }
    sycl::buffer<floatT, 1> dm_buf(dm_host_ptr, sycl::range<1>(n_train * n_test));

    auto res_ev =
        exec_q.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(depends);
                sycl::accessor<floatT, 3, sycl::access::mode::read_write, sycl::access::target::local>
                    scratch(sycl::range<3>(n_train_chunk, n_test_chunk, dim_chunk), cgh);
                sycl::accessor dm_acc(dm_buf, cgh, sycl::read_write);
                sycl::accessor X_train_acc(X_train_buf, cgh, sycl::read_only);
                sycl::accessor X_test_acc(X_test_buf, cgh, sycl::read_only);

                auto ndRange = sycl::nd_range<3>(
                    sycl::range<3>(
                        n_train_chunk * ceiling_quotient(n_train, static_cast<size_t>(n_train_chunk)),
                        n_test_chunk * ceiling_quotient(n_test, static_cast<size_t>(n_test_chunk)),
                        dim_chunk),
                    sycl::range<3>(n_train_chunk, n_test_chunk, dim_chunk));
                auto kern_func = [=](sycl::nd_item<3> it)
                {
                    auto i_train = it.get_global_id(0);
                    auto lid_train = it.get_local_id(0);
                    auto i_test = it.get_global_id(1);
                    auto lid_test = it.get_local_id(1);
                    auto lid = it.get_local_id(2);

                    if ((lid < dim) && (i_train < n_train) && (i_test < n_test))
                    {
                        floatT sum_sq(0);
                        for (size_t k = lid; k < dim; k += dim_chunk)
                        {
                            const floatT diff = (X_train_acc[i_train * dim + k] -
                                                 X_test_acc[i_test * dim + k]);
                            sum_sq += diff * diff;
                        }
                        scratch[sycl::id<3>(lid_train, lid_test, lid)] = sum_sq;
                    }
                    else
                    {
                        scratch[sycl::id<3>(lid_train, lid_test, lid)] = floatT(0);
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (dim_chunk == 2)
                    {
                        int i = dim_chunk >> 1;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    else if constexpr (dim_chunk == 4)
                    {
                        int i = dim_chunk >> 1;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                        i = dim_chunk >> 2;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    else if constexpr (dim_chunk == 8)
                    {
                        int i = dim_chunk >> 1;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                        i = dim_chunk >> 2;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                        i = dim_chunk >> 3;
                        if (lid < static_cast<decltype(lid)>(i))
                            scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    else
                    {
                        for (int i = dim_chunk / 2; i > 0; i >>= 1)
                        {
                            if (lid < static_cast<decltype(lid)>(i))
                                scratch[sycl::id<3>(lid_train, lid_test, lid)] += scratch[sycl::id<3>(lid_train, lid_test, lid + i)];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                    }
                    if (lid == 0 && (i_test < n_test) && (i_train < n_train))
                    {
                        dm_acc[i_test * ld_dm + i_train] = scratch[sycl::id<3>(lid_train, lid_test, 0)];
                    }
                };
                cgh.parallel_for(ndRange, kern_func);
            });

    return res_ev;
}

template <typename floatT>
sycl::event pw_dsq_mkl(
    sycl::queue &exec_q,
    size_t dim,
    floatT *X_train_host_ptr,
    size_t n_train,
    floatT *X_test_host_ptr,
    size_t n_test,
    floatT *dm_host_ptr,
    size_t ld_dm,
    const sycl::vector_class<sycl::event> &depends = {})
{
    sycl::buffer<floatT, 1> X_train_buf{X_train_host_ptr, sycl::range<1>(n_train * dim)};
    sycl::buffer<floatT, 1> X_test_buf{X_test_host_ptr, sycl::range<1>(n_test * dim)};
    sycl::buffer<floatT, 1> dm_buf{dm_host_ptr, sycl::range<1>(n_train * n_test)};

    sycl::buffer<floatT, 1> X_train_l2_buf{sycl::range<1>(n_train)};
    sycl::buffer<floatT, 1> X_test_l2_buf{sycl::range<1>(n_test)};

    oneapi::mkl::blas::row_major::gemm(
        exec_q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n_test, n_train, dim,
        double(2),
        X_test_buf, dim,
        X_train_buf, n_train,
        double(0),
        dm_buf,
        n_train);

    return sycl::event();
}

template <typename floatT>
sycl::event pw_dsq_host2(
    sycl::queue &exec_q,
    size_t dim,
    floatT *X_train_host_ptr,
    size_t n_train,
    floatT *X_test_host_ptr,
    size_t n_test,
    floatT *dm_host_ptr,
    size_t ld_dm,
    const sycl::vector_class<sycl::event> &depends = {})
{
    constexpr int n_train_chunk = 8;
    constexpr int n_test_chunk = 32;

    sycl::buffer<floatT, 1> X_train_buf(X_train_host_ptr, sycl::range<1>(n_train * dim));
    sycl::buffer<floatT, 1> X_test_buf(X_test_host_ptr, sycl::range<1>(n_test * dim));
    sycl::buffer<floatT, 1> dm_buf(dm_host_ptr, sycl::range<1>(n_train * n_test));

    auto res_ev =
        exec_q.submit(
            [&](sycl::handler &cgh)
            {
                cgh.depends_on(depends);
                sycl::accessor dm_acc(dm_buf, cgh, sycl::read_write);
                sycl::accessor X_train_acc(X_train_buf, cgh, sycl::read_only);
                sycl::accessor X_test_acc(X_test_buf, cgh, sycl::read_only);
                sycl::accessor<floatT, 2, sycl::access::mode::read_write, sycl::access::target::local>
                    slm_test(sycl::range<2>(n_test_chunk, dim), cgh);

                auto ndRange = sycl::nd_range<2>(
                    sycl::range<2>(
                        n_train_chunk * ceiling_quotient(n_train, static_cast<size_t>(n_train_chunk)),
                        n_test_chunk * ceiling_quotient(n_test, static_cast<size_t>(n_test_chunk))),
                    sycl::range<2>(n_train_chunk, n_test_chunk));
                auto kern_func = [=](sycl::nd_item<2> it)
                {
                    auto i_train = it.get_global_id(0);
                    auto lid_train = it.get_local_id(0);
                    auto i_test = it.get_global_id(1);
                    auto lid_test = it.get_local_id(1);

                    if (lid_train == 0)
                    {
                        for (size_t elem_id = 0; elem_id < dim; ++elem_id)
                        {
                            slm_test[sycl::id<2>(lid_test, elem_id)] = X_test_acc[i_test * dim + elem_id];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if ((i_train < n_train) && (i_test < n_test))
                    {
                        floatT sum_sq(0);
#if 1
                        for (size_t k = 0; k < dim; k += 4)
                        {
                            sycl::vec<floatT, 4> train_v, test_v;
                            train_v.load(0, X_train_acc.get_pointer() + i_train * dim + k);
                            test_v.load(0, slm_test.get_pointer() + lid_test * dim + k);
                            const auto diff = train_v - test_v;
                            auto diff_sq = diff * diff;
                            sum_sq += diff_sq[0] +
                                      ((k + 1 < dim) ? diff_sq[1] : floatT(0)) +
                                      ((k + 2 < dim) ? diff_sq[2] : floatT(0)) +
                                      ((k + 3 < dim) ? diff_sq[3] : floatT(0));
                        }
#endif
#if 0
			constexpr size_t k_step = 8;
                        for (size_t k = 0; k < dim; k += k_step)
                        {
                            sycl::vec<floatT, k_step> train_v, test_v;
                            train_v.load(0, X_train_acc.get_pointer() + i_train * dim + k);
                            test_v.load(0, slm_test.get_pointer() + lid_test * dim + k);
                            const auto diff = train_v - test_v;
                            auto diff_sq = diff * diff;
                            sum_sq += (
				diff_sq[0] +
				((k + 1 < dim) ? diff_sq[1] : floatT(0)) +
			        ((k + 2 < dim) ? diff_sq[2] : floatT(0)) +
			        ((k + 3 < dim) ? diff_sq[3] : floatT(0)) +
			        ((k + 4 < dim) ? diff_sq[4] : floatT(0)) +
			        ((k + 5 < dim) ? diff_sq[5] : floatT(0)) +
			        ((k + 6 < dim) ? diff_sq[6] : floatT(0)) +
			        ((k + 7 < dim) ? diff_sq[7] : floatT(0))
				);
                        }
#endif
#if 0
                        for (size_t k = 0; k < dim; k += 2)
                        {
                            sycl::vec<floatT, 4> train_v, test_v;
                            train_v.load(0, X_train_acc.get_pointer() + i_train * dim + k);
                            test_v.load(0, X_test_acc.get_pointer() + i_test * dim + k);
                            const auto diff = train_v - test_v;
                            auto diff_sq = diff * diff;
                            sum_sq += diff_sq[0] +
			      ((k + 1 < dim) ? diff_sq[1] : floatT(0));
                        }
#endif
                        dm_acc[i_test * ld_dm + i_train] = sum_sq;
                    }
                };
                cgh.parallel_for(ndRange, kern_func);
            });

    return res_ev;
}

static inline double elapsed(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-06;
}

int main(void)
{
    sycl::queue q;
    size_t n_train = 8 * 4096;
    size_t dim = 25;
    size_t n_test = 512;

    double *X_train_host = new double[n_train * dim];
    double *X_test_host = new double[n_test * dim];

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

    double *X_train_usm = sycl::malloc_device<double>(dim * n_train, q);
    double *X_test_usm = sycl::malloc_device<double>(dim * n_test, q);
    double *dm_usm = sycl::malloc_device<double>(n_test * n_train, q); // matrix (n_test, n_train), C-contiguous

    sycl::event copy_train_ev = q.memcpy(X_train_usm, X_train_host, dim * n_train * sizeof(double));
    sycl::event copy_test_ev = q.memcpy(X_test_usm, X_test_host, dim * n_test * sizeof(double));

    for (size_t k = 0; k < 3; ++k)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto pw_ev = pw_dsq_usm<double>(q, dim, X_train_usm, n_train, X_test_usm, n_test, dm_usm, n_train, {copy_train_ev, copy_test_ev});
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "[pw_dsq_usm] Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;
    }

    std::cout << "============================" << std::endl;

    double *dm_host = new double[n_test * n_train];

    for (size_t k = 0; k < 3; ++k)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto pw_ev = pw_dsq_host<double>(q, dim, X_train_host, n_train, X_test_host, n_test, dm_host, n_train);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "[pw_dsq_host] Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;
    }

    std::cout << "============================" << std::endl;

    for (size_t k = 0; k < 3; ++k)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto pw_ev = pw_dsq_host2<double>(q, dim, X_train_host, n_train, X_test_host, n_test, dm_host, n_train);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "[pw_dsq_host2] Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;
    }

    std::cout << "============================" << std::endl;

    double *dm_usm_copy = new double[n_test * n_train];

    q.memcpy(dm_usm_copy, dm_usm, n_train * n_test * sizeof(double)).wait();
    bool passed = true;
    double max_discr(0);
    for (size_t i = 0; i < n_test * n_train; ++i)
    {
        auto discr = std::fabs(dm_usm_copy[i] - dm_host[i]);
        passed = passed && (discr < double(1e-10));
        if (discr > max_discr)
        {
            max_discr = discr;
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    if (passed)
    {
        std::cout << "Passed" << std::endl;
    }
    else
    {
        std::cout << "Failed, max_discr = " << max_discr << std::endl;
    }

    for (size_t k = 0; k < 3; ++k)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto pw_ev = pw_dsq_mkl<double>(q, dim, X_train_host, n_train, X_test_host, n_test, dm_host, n_train);
        q.wait();
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "Elapsed parallel_knn wall-time: " << elapsed(t0, t1) << " microseconds." << std::endl;
    }

    delete[] dm_host;
    delete[] dm_usm_copy;

    delete[] X_train_host;
    delete[] X_test_host;

    sycl::free(dm_usm, q);
    sycl::free(X_test_usm, q);
    sycl::free(X_train_usm, q);

    return 0;
}
