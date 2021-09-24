import pdist_aggregation_sycl as pas, numpy as np, dpctl
def distance_squared_matrix_np(X1, X2):
    return np.array([
        [
            np.sum(np.square(v1 - v2)) for v1 in X1
        ] for v2 in X2
    ])

q = dpctl.SyclQueue(property="enable_profiling")
t = dpctl.SyclTimer()

X1, X2 = np.random.randn(1004, 45), np.random.randn(2007, 45)

print("Considering distances between shapes:", X1.shape, "and", X2.shape)
print("Using Sycl device:", q.sycl_device.name)

with t(q):  m_sycl = pas.distance_squared_matrix(X1, X2, q)
print("SYCL computation:", t.dt)
with t(q): m_ref = distance_squared_matrix_np(X1, X2)
print("Reference computation:", t.dt)
print("Result agree?", np.allclose(m_sycl, m_ref))
