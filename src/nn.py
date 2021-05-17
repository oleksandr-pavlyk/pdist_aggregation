from sklearn.utils.validation import check_array

from pdist_aggregation import parallel_knn


class NearestNeighbors:
    def __init__(self, n_neighbors=1, algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.X_ = X
        return self

    def kneighbors(self, X, chunk_size=4096, return_distance=False):
        X = check_array(X, order="C")
        return parallel_knn(X, self.X_, k=self.n_neighbors, chunk_size=chunk_size)
