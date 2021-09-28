cd ../ &&
git clone --recursive https://github.com/CEED/libCEED.git &&
cd libCEED && git checkout v0.9.0 && make configure CUDA_DIR=${CUDA_HOME} && make -j &&
cd ../ &&
git clone --recursive https://github.com/hypre-space/hypre.git &&
cd hypre && git checkout v2.22.1 && cd src/ && env HYPRE_CUDA_SM=70 ./configure CC=mpicc CXX=mpicxx --with-cuda  && make -j 16 &&
cd ../../ &&
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz &&
tar -zxvf metis-4.0.3.tar.gz && cd metis-4.0.3 &&
make -j 3 && cd .. &&
ln -s metis-4.0.3 metis-4.0 &&
git clone --recursive https://github.com/NVIDIA/AMGX.git amgx && cd amgx &&
mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../ ../ && make -j all && make install &&
cd ../../ &&
cd mfem && make pcudebug MFEM_USE_AMGX=YES MFEM_USE_CEED=YES CUDA_ARCH=sm_70 -j
