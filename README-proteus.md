# Building MFEM with Proteus

These instruction are for the Tioga machine.
They will need modifications on other machines (e.g., `HIP_ARCH`).
We assume ROCm is loaded in the environment (e.g., `ml rocm/6.2.1`).

1. See MFEM building documentation on how to install prerequisites (HYPRE, METIS, etc.)

2. Clone proteus to your base directory
```bash
git clone https://github.com/Olympus-HPC/proteus.git
```

3. Build and install proteus
```bash
cmake -S proteus -B build-proteus \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_INSTALL_DIR=${ROCM_PATH}/llvm \
-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
-DPROTEUS_ENABLE_HIP=on \ 
-DENABLE_TESTS=off \
-DCMAKE_INSTALL_PREFIX=<proteus-install-path>

cd build-proteus
make -l install
cd ..
```

4. Configure and build MFEM with proteus
```bash
cd mfem
cmake -S . -B build-with-proteus \
-DCMAKE_CXX_COMPILER=amdclang++ \
-DMFEM_USE_HIP=on \
-DHIP_ARCH=gfx90a \
-DMFEM_USE_PROTEUS=on \
-Dproteus_DIR=<proteus-install-path>

cd build-with-proteus
make -j
```
