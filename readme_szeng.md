## install MFEM on XPS15 WSL ubuntu22.04

- Release
```sh
mkdir build-wsl
cd build-wsl
cmake .. -DScaLAPACK_DIR='/usr/local/lib/cmake/scalapack-2.1.0' -DSCALAPACK_LIB='/home/shubin/opt/scalapack/lib/libscalapack.a' -DMKL_CPARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_CPARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_DIRS=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_LIBRARIES='/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.a;' -DMKL_PARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMFEM_USE_SUITESPARSE=YES -DCMAKE_BUILD_TYPE=Release
```

- Debug
```sh
mkdir build-wsl
cd build-wsl
cmake .. -DScaLAPACK_DIR='/usr/local/lib/cmake/scalapack-2.1.0' -DSCALAPACK_LIB='/home/shubin/opt/scalapack/lib/libscalapack.a' -DMKL_CPARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_CPARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_DIRS=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_LIBRARIES='/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_gnu_thread.so' -DMKL_CPARDISO_MKL_MPI_WRAPPER_LIBRARY=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_blacs_openmpi_ilp64.so -DMKL_PARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMFEM_USE_SUITESPARSE=YES -DCMAKE_BUILD_TYPE=Debug

#or 
cmake .. -DScaLAPACK_DIR='/usr/local/lib/cmake/scalapack-2.1.0' -DSCALAPACK_LIB='/home/shubin/opt/scalapack/lib/libscalapack.a' -DMFEM_USE_SUITESPARSE=YES -DMKL_CPARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_CPARDISO_MKL_MPI_WRAPPER_LIBRARY=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so -DMKL_CPARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DMKL_CPARDISO_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DCMAKE_BUILD_TYPE=Debug
```

Detailed build command for `make ex11p`
```sh
cd /home/shubin/01_coding/01_cpp/mfem/build-wsl/examples && /usr/bin/cmake -E cmake_link_script CMakeFiles/ex11p.dir/link.txt --verbose=1
```

```sh
# this is the correct one to link MKL Pardiso, Do Not Delete
/usr/bin/c++ -g CMakeFiles/ex99_test_PML.dir/ex99_test_PML.cpp.o -o ex99_test_PML  -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/lib/x86_64-linux-gnu/openmpi/lib::::::::::::::: ../libmfem.a -lgfortran -lgomp /home/shubin/opt/hypre/src/hypre/lib/libHYPRE.a /usr/lib/x86_64-linux-gnu/libumfpack.so /usr/lib/x86_64-linux-gnu/libklu.so /usr/lib/x86_64-linux-gnu/libamd.so /usr/lib/x86_64-linux-gnu/libbtf.so /usr/lib/x86_64-linux-gnu/libcholmod.so /usr/lib/x86_64-linux-gnu/libcolamd.so /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /home/shubin/opt/mumps/lib/libdmumps.a /home/shubin/opt/mumps/lib/libmumps_common.a /home/shubin/opt/mumps/lib/libpord.a /usr/lib/x86_64-linux-gnu/libmpi_usempif08.so /usr/lib/x86_64-linux-gnu/libmpi_usempi_ignore_tkr.so /usr/lib/x86_64-linux-gnu/libmpi_mpifh.so /usr/lib/x86_64-linux-gnu/libopen-rte.so /usr/lib/x86_64-linux-gnu/libopen-pal.so /usr/lib/x86_64-linux-gnu/libhwloc.so /usr/lib/x86_64-linux-gnu/libevent_core.so /usr/lib/x86_64-linux-gnu/libevent_pthreads.so -lm /usr/lib/x86_64-linux-gnu/libz.so /home/shubin/opt/parmetis/lib/libparmetis.a /home/shubin/opt/metis-5.1.0/lib/libmetis.a /usr/local/lib/libscalapack.a /usr/lib/x86_64-linux-gnu/libopenblas.so -lm -ldl /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so -lgomp -lpthread -lm -ldl /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so 
```
