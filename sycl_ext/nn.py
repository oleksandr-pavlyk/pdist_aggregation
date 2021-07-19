import abc

import dpctl
from sklearn.utils.validation import check_array
from pdist_aggregation_sycl import parallel_knn


class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="sycl", queue=None):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors
        if queue is None:
            self.queue = dpctl.SyclQueue()
        elif isinstance(queue, dpctl.SyclQueue):
            self.queue = queue
        else:
            self.queue = dpctl.SyclQueue(queue)

    def fit(self, X):
        self._X_train = check_array(X, order="C")
        return self

    def kneighbors(self, X, return_distance=False):
        X = check_array(X, order="C")
        return parallel_knn(
            self._X_train,
            X,
            k=self.n_neighbors,
            queue=self.queue,
            strategy="auto",
            return_distance=return_distance,
        )
