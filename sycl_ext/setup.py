import numpy
from Cython.Build import build_ext
from setuptools import Extension, setup
import dpctl

extensions = [
    Extension(
        name="pdist_aggregation_sycl",
        sources=["_pdist_aggregation_sycl.pyx"],
        depends=["parallel_knn_helper.hpp", "parallel_knn.hpp", "heap.hpp"],
        include_dirs=[numpy.get_include(), dpctl.get_include()],
        extra_compile_args=["-Wall", "-O2", "-Wextra", "-fsycl", "-fsycl-unnamed-lambda", "-Wno-pass-failed"],
        extra_link_args=["-fPIC"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++"
    )
]

setup(
    name="pdist_aggregation_sycl",
    cmdclass={"build_ext": build_ext},
    version="0.1",
    ext_modules=extensions,
    install_requires=["setuptools>=18.0", "cython>=0.29", "numpy"],
    python_requires=">=3.7",
)
