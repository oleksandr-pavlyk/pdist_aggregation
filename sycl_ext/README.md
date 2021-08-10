# Python extension solvign k-NN problem using Intel(R) oneAPI DPC++

```
(idp) [14:57:13 ansatnuc04 sycl_ext]$ python run.py
Solving KNN problem:
   n_train = 1048576
   dim = 5
   n_test = 273
   k = 5

GPU device:
    Name            Intel(R) UHD Graphics [0x9bca]
    Driver version  1.1.20389
    Vendor          Intel(R) Corporation
    Profile         FULL_PROFILE
    Filter string   level_zero:gpu:0
CPU device:
    Name            Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz
    Driver version  2021.12.6.0.19_160000
    Vendor          Intel(R) Corporation
    Profile         FULL_PROFILE
    Filter string   opencl:cpu:0

SYCL computed indexes agreed with reference computed indexes
Reference result computation took 36.34312146436423 seconds
SYCL algo on GPU computation took 1.2743729911744595 seconds
SYCL algo on CPU computation took 0.8498784499242902 seconds

Repeating SYCL algo runs (JIT-ting has already happened)
SYCL algo on GPU computation took 0.8518184367567301 seconds
SYCL algo on CPU computation took 0.8513380466029048 seconds
```

# To build the extension:

```
cd sycl_ext
# activate DPC++ compiler
. /opt/intel/oneapi/compiler/latest/env/vars.sh
CC=clang CXX=dpcpp python setup.py build_ext --inplace
python run.py
```

[Profiling tools interfaces for GPU](https://github.com/intel/pti-gpu) was instrumental in improving performance,
specifically `onetrace` and `ze_hot_kernels` utilities.

Other useful resources were
   - Data Parallel C++ book, [freely accessible from Apress](https://www.apress.com/gp/book/9781484255735).
   - Intel(R) GPU optimization [guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html)
   - Intel(R) Vtune's [GPU analysis](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/code-profiling-scenarios/gpu-application-analysis.html)
   - Intel(R) Advisor with [GPU roofline](https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-advisor/top.html).