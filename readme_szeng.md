## install MFEM on XPS15 WSL ubuntu22.04
```sh
mkdir build-wsl
cd build-wsl
cmake .. -DScaLAPACK_DIR='/usr/local/lib/cmake/scalapack-2.1.0' -DSCALAPACK_LIB='/home/shubin/opt/scalapack/lib/libscalapack.a' -DMKL_CPARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_CPARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DCMAKE_BUILD_TYPE=Release
```