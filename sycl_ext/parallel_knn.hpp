#pragma once

#include <CL/sycl.hpp>
#include <limits>
#include "heap.hpp"

template <typename floatT, typename intT>
struct DistanceIndex
{
    floatT dist_;
    intT idx_;

    DistanceIndex() : dist_(std::numeric_limits<floatT>::infinity()), idx_(-1){};
    ~DistanceIndex(){};

    void set(const floatT d, const intT i) noexcept
    {
        dist_ = d;
        idx_ = i;
    }

    floatT get_distance() const noexcept { return dist_; }
    intT get_index() const noexcept { return idx_; }
};

template <typename accT>
class HeapProxy3D
{
    accT acc_;
    size_t ix_;
    size_t iy_;

public:
    HeapProxy3D(accT acc, size_t ix, size_t iy) : acc_(acc), ix_(ix), iy_(iy) {}
    ~HeapProxy3D() {}

    typename accT::reference operator[](size_t k) { return acc_[sycl::id<3>(ix_, iy_, k)]; }
};

template <typename accT>
class HeapProxy2D
{
    accT acc_;
    size_t ix_;

public:
    HeapProxy2D(accT acc, size_t ix) : acc_(acc), ix_(ix) {}
    ~HeapProxy2D() {}

    typename accT::reference operator[](size_t k) { return acc_[sycl::id<2>(ix_, k)]; }
};

template <typename T>
inline T ceiling_quotient(T n, T m)
{
    return ((n + m - 1) / m);
}

template <typename floatT, typename intT>
void parallel_knn_search(
    sycl::queue &exec_q,
    size_t k,
    size_t dim,
    const floatT *X_train_ptr, // host ptr to (n_train_, dim) C-contiguous matrix
    size_t n_train_,
    const floatT *X_test_ptr, // host ptr to (n_test_, dim) C-contiguous matrix
    size_t n_test_,
    intT *knn_indices_ptr,     // host ptr to (n_test_, k) C-contiguous matrix
    floatT *knn_distances_ptr, // host ptr to (n_test_, k) C-contiguous matrix
    const sycl::vector_class<sycl::event> &depends = {})
{
    using pair_t = DistanceIndex<floatT, intT>;
    auto comp = [](pair_t p1, pair_t p2)
    {
        return p1.get_distance() < p2.get_distance();
    };

    const size_t train_chunk_size = std::min(n_train_, std::max(k, size_t(4096)));
    const size_t test_chunk_size = std::max(n_test_, size_t(512));

    sycl::buffer<floatT, 2> dist_buf(
        sycl::range<2>(test_chunk_size, train_chunk_size));
    sycl::buffer<pair_t, 2> aggregation_heap_buf(sycl::range<2>(test_chunk_size, k));

    const size_t n_train_chunks = ceiling_quotient(n_train_, train_chunk_size);
    const size_t n_test_chunks = ceiling_quotient(n_test_, test_chunk_size);

    sycl::buffer<floatT, 1> X_train_buf(X_train_ptr, dim * n_train_, {sycl::property::buffer::use_host_ptr()});
    sycl::buffer<floatT, 1> X_test_buf(X_test_ptr, dim * n_test_, {sycl::property::buffer::use_host_ptr()});

    for (size_t n_test_chunk_id = 0; n_test_chunk_id < n_test_chunks; ++n_test_chunk_id)
    {
        size_t n_test_disp = n_test_chunk_id * test_chunk_size;
        size_t n_test = ((n_test_disp + test_chunk_size) < n_test_) ? test_chunk_size : (n_test_ - n_test_disp);

        exec_q.submit(
            [&](sycl::handler &cgh)
            {
                sycl::accessor aggregation_heap_acc(aggregation_heap_buf, cgh);
                cgh.parallel_for(
                    sycl::range<2>(n_test, k),
                    [=](sycl::item<2> it)
                    {
                        aggregation_heap_acc[it.get_id()] = pair_t();
                    });
            });

        sycl::event prev_iter_dep{};

        for (size_t n_train_chunk_id = 0; n_train_chunk_id < n_train_chunks; ++n_train_chunk_id)
        {
            size_t n_train_disp = n_train_chunk_id * train_chunk_size;
            size_t n_train = (n_train_disp + train_chunk_size < n_train_) ? train_chunk_size : (n_train_ - n_train_disp);
            // no need to track the returned event,
            // as dependency is handled via accessors
            exec_q.submit(
                [&](sycl::handler &cgh)
                {
                    // this task is independent of events given in depends.
                    sycl::accessor dist_acc(dist_buf, cgh, sycl::write_only, sycl::noinit);
                    cgh.fill<floatT>(dist_acc, floatT(0));
                });

            sycl::event pw_dist_ev = exec_q.submit(
                [&](sycl::handler &cgh)
                {
                    int dim_chunk = 16;
                    assert(0 == (dim_chunk & (dim_chunk - 1)));
                    cgh.depends_on(depends);
                    cgh.depends_on(prev_iter_dep);

                    sycl::accessor dist_acc(dist_buf, cgh, sycl::read_write);
                    sycl::accessor<floatT, 1, sycl::access::mode::read_write, sycl::access::target::local>
                        scratch(dim_chunk, cgh);
                    sycl::accessor X_train_acc(X_train_buf, cgh, sycl::read_only);
                    sycl::accessor X_test_acc(X_test_buf, cgh, sycl::read_only);

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(dim_chunk * ceiling_quotient<size_t>(dim, dim_chunk), n_train, n_test),
                            sycl::range<3>(dim_chunk, 1, 1)),
                        [=](sycl::nd_item<3> it)
                        {
                            auto gid = it.get_global_id(0);
                            auto lid = it.get_local_id(0);
                            auto i_train = it.get_global_id(1);
                            auto i_test = it.get_global_id(2);

                            if (gid < dim)
                            {
                                size_t abs_i_train = n_train_disp + i_train;
                                size_t abs_i_test = n_test_disp + i_test;
                                const floatT diff = (X_train_acc[abs_i_train * dim + gid] -
                                                     X_test_acc[abs_i_test * dim + gid]);
                                scratch[lid] = diff * diff;
                            }
                            else
                            {
                                scratch[lid] = floatT(0);
                            }
                            for (int i = dim_chunk / 2; i > 0; i >>= 1)
                            {
                                it.barrier(sycl::access::fence_space::local_space);
                                if (lid < static_cast<decltype(lid)>(i))
                                    scratch[lid] += scratch[lid + i];
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            if (lid == 0)
                            {
                                auto dij = sycl::ONEAPI::atomic_ref<
                                    floatT,
                                    sycl::ONEAPI::memory_order::relaxed,
                                    sycl::ONEAPI::memory_scope::device,
                                    sycl::access::address_space::global_space>(dist_acc[sycl::id<2>(i_test, i_train)]);
                                dij.fetch_add(scratch[0]);
                            }
                        });
                });

            // to find nearest neighbors, (n_train, m) heaps are maintained,
            // each heap is built by a work-item, based on (n_train / m) distances
            size_t m = std::min(size_t(16), ceiling_quotient(n_train, 16 * k));
            sycl::buffer<pair_t, 3> max_heap_buf(sycl::range<3>(n_test, m, k));

            sycl::event topk_ev = exec_q.submit(
                [&](sycl::handler &cgh)
                {
                    cgh.depends_on(pw_dist_ev);
                    sycl::accessor max_heap_acc(max_heap_buf, cgh);
                    sycl::accessor dist_acc(dist_buf, cgh, sycl::read_only);
                    cgh.parallel_for(
                        sycl::range<2>(m, n_test),
                        [=](sycl::item<2> it)
                        {
                            auto heap_id = it.get_id(0);
                            auto test_id = it.get_id(1);
                            auto distances_chunk = ceiling_quotient(n_train, m);
                            const auto comp = [](pair_t p1, pair_t p2)
                            {
                                return p1.get_distance() < p2.get_distance();
                            };

                            // initialize the heap
                            for (size_t elem_id = 0; elem_id < k; ++elem_id)
                            {
                                // has heap structure by construction at
                                // initialization (all distances +inf, all indices -1)
                                max_heap_acc[sycl::id<3>(test_id, heap_id, elem_id)] = pair_t();
                            }

                            size_t count = 0;
                            for (size_t i_train = heap_id * distances_chunk;
                                 (i_train < n_train) && (count < distances_chunk);
                                 ++count, ++i_train)
                            {
                                pair_t top = max_heap_acc[sycl::id<3>(test_id, heap_id, 0)];
                                pair_t cand;
                                size_t abs_i_train = n_train_disp + i_train;
                                floatT dist_itrain_to_testid = dist_acc[sycl::id<2>(test_id, i_train)];
                                cand.set(dist_itrain_to_testid, abs_i_train);
                                if (comp(cand, top))
                                {
                                    // proxy to represent view with fixed index 0, and index 1.
                                    HeapProxy3D<decltype(max_heap_acc)> heap(max_heap_acc, test_id, heap_id);

                                    // no synchronization is needed, since workitems
                                    // work on disjoint blocks of memory
                                    pop_heap<decltype(heap), pair_t, decltype(comp)>(
                                        heap, k, comp);
                                    heap[k - 1] = cand;
                                    push_heap<decltype(heap), pair_t, decltype(comp)>(
                                        heap, k, comp);
                                }
                            }
                        });
                });
            prev_iter_dep = topk_ev;

            // we now have m heaps, for the each test point in the chunk
            // we aggregate these heaps into aggregation heap on the host.
            {
                sycl::host_accessor max_heap_hacc(max_heap_buf, sycl::read_only);
                sycl::host_accessor aggregation_heap_hacc(aggregation_heap_buf);

                for (size_t i_test = 0; i_test < n_test; ++i_test)
                {
                    HeapProxy2D<decltype(aggregation_heap_hacc)> res_heap(aggregation_heap_hacc, i_test);
                    for (size_t heap_chunk_id = 0; heap_chunk_id < m; ++heap_chunk_id)
                    {
                        for (size_t heap_elem_id = 0; heap_elem_id < k; ++heap_elem_id)
                        {
                            pair_t top = res_heap[0];
                            pair_t cand = max_heap_hacc[sycl::id<3>(i_test, heap_chunk_id, heap_elem_id)];
                            if (comp(cand, top))
                            {
                                pop_heap<decltype(res_heap), pair_t, decltype(comp)>(
                                    res_heap, k, comp);
                                res_heap[k - 1] = cand;
                                push_heap<decltype(res_heap), pair_t, decltype(comp)>(
                                    res_heap, k, comp);
                            }
                        }
                    }
                }
            }
        } // end of loop over training vectors

        {
            sycl::host_accessor aggregation_heap_hacc(aggregation_heap_buf);
            for (size_t i_test = 0; i_test < n_test; ++i_test)
            {

                size_t abs_i_test = n_test_disp + i_test;
                HeapProxy2D<decltype(aggregation_heap_hacc)> res_heap(aggregation_heap_hacc, i_test);
                sort_heap<decltype(res_heap), pair_t, decltype(comp)>(res_heap, k, comp);
                // res_heap is now sorted, from least distance to k-th largest one
                floatT prev = -std::numeric_limits<floatT>::infinity();
                for (size_t i = 0; i < k; ++i)
                {
                    pair_t p = res_heap[i];
                    floatT dist = p.get_distance();
                    assert(prev <= dist);
                    intT ind = p.get_index();
                    knn_indices_ptr[abs_i_test * k + i] = ind;
                    knn_distances_ptr[abs_i_test * k + i] = dist;
                    prev = dist;
                }
            }
        }
    }
    return;
}
