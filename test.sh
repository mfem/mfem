make serial MFEM_USE=SUITESPARSE=YES SUITESPARSE_OPT=-I/usr/include/suitesparse MFEM_USE_LAPACK=YES LAPACK_LIB='-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -L/sfw/intel/2024.0.1/compiler/latest/lib -liomp5 -lpthread' -j 11
cd examples
make ex38
./ex38 -i volumetric3d -r 3