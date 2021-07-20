import numpy as np, nn
from timeit import default_timer

np.random.seed(1234)
X_train = np.random.randn(1024*1024, 5)
X_test = np.random.randn(273, 5)

k = 5
print("Solving KNN problem:")
print("   n_train = {}".format(X_train.shape[0]))
print("   dim = {}".format(X_train.shape[1]))
print("   n_test = {}".format(X_test.shape[0]))
print("   k = {}".format(k))
print(" ")

knn = nn.NearestNeighborsParallelAuto(n_neighbors=k)
knn.fit(X_train)

print("")
t0 = default_timer()
ind_ref = np.asarray([ np.argsort(np.square(X_train - v).sum(axis=-1))[:k] for v in X_test ])
t1 = default_timer()
ind_pdist, _ = knn.kneighbors(X_test)
t2 = default_timer()

assert np.all(np.equal(ind_ref, ind_pdist)), "PDIST result  disagrees with reference"

print("PDIST computed indexes agreed with reference computed indexes")

print("Reference result computation took {} seconds".format(t1-t0))
print("PDIST brut-force algo computation took {} seconds".format(t2-t1))
