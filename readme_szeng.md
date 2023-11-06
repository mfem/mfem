## install MFEM on XPS15 WSL ubuntu22.04
```sh
mkdir build-wsl
cd build-wsl
cmake .. -DScaLAPACK_DIR='/usr/local/lib/cmake/scalapack-2.1.0' -DSCALAPACK_LIB='/home/shubin/opt/scalapack/lib/libscalapack.a' -DMKL_CPARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_CPARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_DIRS=/opt/intel/oneapi/mkl/latest/lib/intel64 -DMKL_PARDISO_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include -DMKL_PARDISO_LIBRARIES='/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a;/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.a;' -DMKL_PARDISO_INCLUDE_DIRS=/opt/intel/oneapi/mkl/latest/include -DMKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64 -DCMAKE_BUILD_TYPE=Release
```

Detailed build command for `make ex11p`
```sh
cd /home/shubin/01_coding/01_cpp/mfem/build-wsl/examples && /usr/bin/cmake -E cmake_link_script CMakeFiles/ex11p.dir/link.txt --verbose=1
```

```sh
/usr/bin/c++ -O3 -DNDEBUG CMakeFiles/ex25p.dir/ex25p.cpp.o -o ex25p  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib::::::::::::::: ../libmfem.a -lgfortran -lgomp /home/shubin/opt/hypre/src/hypre/lib/libHYPRE.a /home/shubin/opt/mumps/lib/libdmumps.a /home/shubin/opt/mumps/lib/libmumps_common.a /home/shubin/opt/mumps/lib/libpord.a /usr/lib/x86_64-linux-gnu/libmpi_usempif08.so /usr/lib/x86_64-linux-gnu/libmpi_usempi_ignore_tkr.so /usr/lib/x86_64-linux-gnu/libmpi_mpifh.so /usr/lib/x86_64-linux-gnu/libopen-rte.so /usr/lib/x86_64-linux-gnu/libopen-pal.so /usr/lib/x86_64-linux-gnu/libhwloc.so /usr/lib/x86_64-linux-gnu/libevent_core.so /usr/lib/x86_64-linux-gnu/libevent_pthreads.so -lm /usr/lib/x86_64-linux-gnu/libz.so /home/shubin/opt/parmetis/lib/libparmetis.a /home/shubin/opt/metis-5.1.0/lib/libmetis.a /usr/local/lib/libscalapack.a /usr/lib/x86_64-linux-gnu/libopenblas.so -lm -ldl /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.a /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so -lgomp -lgfortran
```

```sh
/usr/bin/c++ -O3 -DNDEBUG CMakeFiles/ex99_test_PML.dir/ex99_test_PML.cpp.o -o ex99_test_PML  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib::::::::::::::: ../libmfem.a -lgfortran -lgomp /home/shubin/opt/hypre/src/hypre/lib/libHYPRE.a /home/shubin/opt/mumps/lib/libdmumps.a /home/shubin/opt/mumps/lib/libmumps_common.a /home/shubin/opt/mumps/lib/libpord.a /usr/lib/x86_64-linux-gnu/libmpi_usempif08.so /usr/lib/x86_64-linux-gnu/libmpi_usempi_ignore_tkr.so /usr/lib/x86_64-linux-gnu/libmpi_mpifh.so /usr/lib/x86_64-linux-gnu/libopen-rte.so /usr/lib/x86_64-linux-gnu/libopen-pal.so /usr/lib/x86_64-linux-gnu/libhwloc.so /usr/lib/x86_64-linux-gnu/libevent_core.so /usr/lib/x86_64-linux-gnu/libevent_pthreads.so -lm /usr/lib/x86_64-linux-gnu/libz.so /home/shubin/opt/parmetis/lib/libparmetis.a /home/shubin/opt/metis-5.1.0/lib/libmetis.a /usr/local/lib/libscalapack.a /usr/lib/x86_64-linux-gnu/libopenblas.so -lm -ldl /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.a /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
```

```sh
export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_gnu_thread.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so
```
