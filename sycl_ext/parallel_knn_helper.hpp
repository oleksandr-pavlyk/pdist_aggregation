#pragma once

#include <CL/sycl.hpp>
#include "parallel_knn.hpp"
#include "dpctl_sycl_types.h"

template<typename floatT, typename intT>
void p_knn_search(
    DPCTLSyclQueueRef QRef,
    size_t k, size_t dim,
    const floatT *X_train_ptr,
    size_t n_train,
    const floatT* X_test_ptr,
    size_t n_test,
    intT* knn_indices_ptr,
    floatT *knn_distances_ptr
    )
{
    sycl::queue execution_queue = *(reinterpret_cast<sycl::queue *>(QRef));

    return parallel_knn_search(
	execution_queue, k, dim, X_train_ptr, n_train, X_test_ptr, n_test, knn_indices_ptr, knn_distances_ptr);
}


template<typename floatT>
void pw_d2m(
    DPCTLSyclQueueRef QRef,
    size_t dim,
    const floatT *X_train_ptr,
    size_t n_train,
    const floatT* X_test_ptr,
    size_t n_test,
    floatT *dsqM
    )
{
    sycl::queue execution_queue = *(reinterpret_cast<sycl::queue *>(QRef));

    {
	sycl::buffer<floatT, 1> X_train_buf(X_train_ptr, dim * n_train, {sycl::property::buffer::use_host_ptr()});
	sycl::buffer<floatT, 1> X_test_buf(X_test_ptr, dim * n_test, {sycl::property::buffer::use_host_ptr()});
	sycl::buffer<floatT, 2> dist_buf(dsqM, sycl::range<2>(n_test, n_train), {sycl::property::buffer::use_host_ptr()});

	sycl::event ev;
	compute_pw_dist_opt2<floatT>(
	    execution_queue,
            dim,
	    X_train_buf, n_train,
	    X_test_buf, n_test,
	    dist_buf,
	    ev);
    }

    return;
}
