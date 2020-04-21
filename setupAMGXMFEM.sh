cd ../ &&
wget https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.10.0b.tar.gz &&
tar -zxvf hypre-2.10.0b.tar.gz &&
mv hypre-2.10.0b hypre &&
cd hypre/src/ && mkdir build && cd build &&
cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../../ ../ && make -j && make install &&
cd ../../../ &&
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz &&
tar -zxvf metis-4.0.3.tar.gz && cd metis-4.0.3 &&
make -j 3 && cd .. &&
ln -s metis-4.0.3 metis-4.0 &&
git clone --recursive git@github.com:NVIDIA/AMGX.git && cd AMGX &&
mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/../ ../ && make -j all && make install && cd ../../ &&
cd mfem && make pcuda CUDA_ARCH=sm_70 MFEM_USE_AMGX=YES -j
