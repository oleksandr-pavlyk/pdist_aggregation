import numpy as np, dpctl, nn
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

gpu_queue = dpctl.SyclQueue("gpu")
print("GPU device:")
gpu_queue.sycl_device.print_device_info()
cpu_queue = dpctl.SyclQueue("cpu")
print("CPU device:")
cpu_queue.sycl_device.print_device_info()

knn_gpu = nn.NearestNeighbors(n_neighbors=k, queue=gpu_queue)
knn_gpu.fit(X_train)

knn_cpu = nn.NearestNeighbors(n_neighbors=k, queue=cpu_queue)
knn_cpu.fit(X_train)

print("")
t0 = default_timer()
ind_ref = np.asarray([ np.argsort(np.square(X_train - v).sum(axis=-1))[:k] for v in X_test ])
t1 = default_timer()
ind_gpu = knn_gpu.kneighbors(X_test)
t2 = default_timer()
ind_cpu = knn_gpu.kneighbors(X_test)
t3 = default_timer()

assert np.array_equal(ind_ref, ind_gpu), "SYCL result computed on GPU disagrees with reference"
assert np.array_equal(ind_ref, ind_cpu), "SYCL result computed on GPU disagrees with reference"

print("SYCL computed indexes agreed with reference computed indexes")

print("Reference result computation took {} seconds".format(t1-t0))
print("SYCL algo on GPU computation took {} seconds".format(t2-t1))
print("SYCL algo on CPU computation took {} seconds".format(t3-t2))


print("")
print("Repeating SYCL algo runs (JIT-ting has already happened)")
t1 = default_timer()
ind_gpu = knn_gpu.kneighbors(X_test)
t2 = default_timer()
ind_cpu = knn_gpu.kneighbors(X_test)
t3 = default_timer()
print("SYCL algo on GPU computation took {} seconds".format(t2-t1))
print("SYCL algo on CPU computation took {} seconds".format(t3-t2))
