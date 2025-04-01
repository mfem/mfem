export LC_USER=andrej1
module load rocmcc/6.3.1-cce-19.0.0-magic cmake/3.29.2

export MPICH_CC=amdclang
export MPICH_CXX=amdclang++
export ROCM_PATH=/opt/rocm-6.3.1
export LLVM_DIR=$ROCM_PATH/lib/llvm
export MPI_DIR=/usr/tce/packages/cray-mpich/cray-mpich-8.1.32-rocmcc-6.3.1-cce-19.0.0-magic

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$ROCM_PATH/lib/cmake/hip:$ROCM_PATH/lib/cmake/hipblas:$ROCM_PATH/lib/cmake/hipblas-common:$ROCM_PATH/lib/cmake/hipsparse:$ROCM_PATH/lib/cmake/rocsparse:$ROCM_PATH/lib/cmake/rocrand

export BASE_DIR=/usr/workspace/$LC_USER/dfem-tuo-magic
export LOCAL_DIR=/usr/workspace/$LC_USER/dfem-tuo-magic/local
mkdir -p $LOCAL_DIR
export PATH=$LOCAL_DIR/bin:$PATH
cd $BASE_DIR

## Enzyme
git clone --depth 1 https://github.com/EnzymeAD/Enzyme.git
pushd Enzyme/enzyme
CC=amdclang CXX=amdclang++ cmake -B build -DLLVM_DIR=$LLVM_DIR -DCMAKE_INSTALL_PREFIX=$LOCAL_DIR
cmake --build build -j && cmake --install build
popd

## hypre
curl https://github.com/hypre-space/hypre/archive/refs/tags/v2.32.0.tar.gz -o hypre-v2.32.0.tar.gz -L
tar xzf hypre-v2.32.0.tar.gz
pushd hypre-2.32.0/src
CC=mpicc CXX=mpicxx CXXFLAGS="std=c++17 -fPIC" CFLAGS="-fPIC" ROCM_PATH=$ROCM_PATH ./configure --disable-fortran --prefix=$LOCAL_DIR --with-MPI-libs="mpi mpich" --with-MPI-lib-dirs=$MPI_DIR/lib --with-MPI-include=$MPI_DIR/include --enable-shared --with-hip
make -j install
popd

## metis
curl -OL https://github.com/mfem/tpls/raw/gh-pages/parmetis-4.0.3.tar.gz
tar xzf parmetis-4.0.3.tar.gz
pushd parmetis-4.0.3
cmake -B build -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC" -DGKLIB_PATH=$BASE_DIR/parmetis-4.0.3/metis/GKlib -DMETIS_PATH=$BASE_DIR/parmetis-4.0.3/metis -DCMAKE_INSTALL_PREFIX=$LOCAL_DIR -DSHARED=1 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j && cmake --install build
popd
pushd parmetis-4.0.3/metis
cmake -B build -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_C_FLAGS="-fPIC" -DGKLIB_PATH=$BASE_DIR/parmetis-4.0.3/metis/GKlib -DCMAKE_INSTALL_PREFIX=$LOCAL_DIR -DSHARED=1 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j && cmake --install build
popd

git clone https://github.com/mfem/mfem.git
git switch dfem-phase1-dev
pushd mfem
CXX=mpicxx cmake -B build-opt -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx942" -DCMAKE_HIP_PLATFORM="amd"
cmake --build build-opt -j
