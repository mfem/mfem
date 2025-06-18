#!/usr/bin/env bash

set -e

# Ensure LLVM_VERSION is set
if [ -z "${LLVM_VERSION}" ]; then
  echo "Error: LLVM_VERSION environment variable is not set"
  exit 1
fi

# Checkout LLVM sources
git clone --filter=blob:none --depth=1 --branch llvmorg-${LLVM_VERSION} --no-checkout https://github.com/llvm/llvm-project.git llvm-project
cd llvm-project
git sparse-checkout set --cone
git checkout llvmorg-${LLVM_VERSION}
git sparse-checkout set cmake llvm/cmake runtimes libcxx libcxxabi
cd ..

mkdir llvm-build && cd llvm-build
cmake -GNinja                                   \
      -DCMAKE_C_COMPILER=${CC}                  \
      -DCMAKE_CXX_COMPILER=${CXX}               \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo         \
      -DCMAKE_INSTALL_PREFIX=/usr               \
      -DLLVM_USE_SANITIZER=${LLVM_SANITIZER}    \
      -DLLVM_BUILD_32_BITS=OFF                  \
      -DLIBCXXABI_USE_LLVM_UNWINDER=OFF         \
      -DLLVM_INCLUDE_TESTS=OFF                  \
      -DLIBCXX_INCLUDE_TESTS=OFF                \
      -DLIBCXX_INCLUDE_BENCHMARKS=OFF           \
      -DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi' \
      ../llvm-project/runtimes/
cmake --build . -- cxx cxxabi
cd ..