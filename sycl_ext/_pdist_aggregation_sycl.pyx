# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as cnp
import dpctl
cimport dpctl as c_dpctl

from cython cimport floating, integral

cdef extern from "parallel_knn_helper.hpp":
    void p_knn_search[floatT, intT](
        c_dpctl.DPCTLSyclQueueRef q,
        size_t k, size_t dim,
        floatT* X_train_ptr, size_t n_train,
        floatT* X_test_ptr, size_t n_test,
        intT *knn_indices_ptr,
        floatT *knn_distance_ptr
    ) nogil

    void pw_d2m[floatT](
        c_dpctl.DPCTLSyclQueueRef q,
        size_t dim,
        floatT* X_train_ptr, size_t n_train,
        floatT* X_test_ptr, size_t n_test,
        floatT* dsq
        ) nogil

def parallel_knn(
        const double[:, ::1] X_train,
        const double[:, ::1] X_test,
        size_t k,
        c_dpctl.SyclQueue queue,
        str strategy = "auto",
        bint return_distance = False):
    int_dtype = np.intp
    float_dtype = np.double
    cdef:
        Py_ssize_t[:, ::1] knn_indices = np.full((X_test.shape[0], k), 0,
                                               dtype=int_dtype)
        double[:, ::1] knn_distances = np.full((X_test.shape[0], k),
                                                  np.inf,
                                                  dtype=float_dtype)
        size_t dim = 0
        size_t n_train = X_train.shape[0]
        size_t n_test = X_test.shape[0]
        c_dpctl.DPCTLSyclQueueRef QRef = NULL

    if strategy != 'auto':
        raise ValueError("Keyword strategy's only supported value is 'auto'")

    if (k == 0):
        raise ValueError("Number requested nearest neighbors must be non-zero")

    if (X_train.shape[1] != X_test.shape[1]):
        raise ValueError("Dimensionality of test and train datasets must be the same")

    dim = X_train.shape[1]

    if (queue is None):
        queue = dpctl.SyclQueue()  # use default

    QRef = queue.get_queue_ref()
    p_knn_search[double, Py_ssize_t](
        QRef, k, dim,
        &X_train[0,0], n_train,
        &X_test[0,0], n_test,
        &knn_indices[0,0],
        &knn_distances[0,0]
    )
    return (np.asarray(knn_indices), np.asarray(knn_distances)) if return_distance else np.asarray(knn_indices)


def distance_squared_matrix(
        const double[:, ::1] X_train,
        const double[:, ::1] X_test,
        c_dpctl.SyclQueue queue
):
    float_dtype = np.double
    cdef:
        size_t dim = 0
        size_t n_train = X_train.shape[0]
        size_t n_test = X_test.shape[0]
        c_dpctl.DPCTLSyclQueueRef QRef = NULL
        double[:, ::1] d2m = np.full((X_test.shape[0], X_train.shape[0]),
                                     np.inf,
                                     dtype=float_dtype)
    if (X_train.shape[1] != X_test.shape[1]):
        raise ValueError("Dimensionality of test and train datasets must be the same")

    dim = X_train.shape[1]

    if (queue is None):
        queue = dpctl.SyclQueue()  # use default

    QRef = queue.get_queue_ref()
    pw_d2m[double](
        QRef, dim, &X_train[0,0], n_train, &X_test[0,0], n_test, &d2m[0,0])

    return np.asarray(d2m)
