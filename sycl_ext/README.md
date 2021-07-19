# Python extension solvign k-NN problem using Intel(R) oneAPI DPC++

```
Solving KNN problem:
   n_train = 1048576
   dim = 5
   n_test = 273
   k = 5

GPU device:
    Name            Intel(R) UHD Graphics [0x9bca]
    Driver version  1.1.20043
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
Reference result computation took 36.348976047709584 seconds
SYCL algo on GPU computation took 9.342610693071038 seconds
SYCL algo on CPU computation took 8.880415248218924 seconds

Repeating SYCL algo runs (JIT-ting has already happenned)
SYCL algo on GPU computation took 8.875004820059985 seconds
SYCL algo on CPU computation took 8.878650922793895 seconds
```

# To build the extension:

```
cd sycl_ext
# activate DPC++ compiler
. /opt/intel/oneapi/compiler/latest/env/vars.sh
CC=clang SYCL=dpcpp python setup.py build_ext --inplace
python run.py
```