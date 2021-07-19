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
