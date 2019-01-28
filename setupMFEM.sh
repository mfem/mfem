make config MFEM_CXX="nvcc" \
CXXFLAGS="-g -Xcompiler -fopenmp -O3 --restrict --expt-extended-lambda -x=cu -arch=sm_60 -std=c++11 -m64" \
MFEM_EXT_LIBS="-L/usr/local/cuda/lib64-lrt -lcuda -lcudart -lcudadevrt -lnvToolsExt -lgomp" \
MFEM_DEBUG=YES OPTIM_FLAGS=-O3 MFEM_USE_MM=YES
