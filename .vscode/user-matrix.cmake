# ----------------------------- Environment Setup -----------------------------
# matrixgcc

# ------------------------------ CMake Commands -------------------------------
# export CMAKE_CFG=Debug            CMAKE_TGT=bench_dfem
# export CMAKE_CFG=Release          CMAKE_TGT=bench_dfem
# export CMAKE_CFG=RelWithDebInfo   CMAKE_TGT=bench_dfem

# ------------------------------- Runs Commands -------------------------------
# ./bench_dfem --benchmark_context=device=gpu --benchmark_filter=BP3/3 
# --benchmark_out_format=csv --benchmark_out=bench_dfem_local_3_matrix.org

# -------------------------------- CMake Setup --------------------------------
cmake_minimum_required(VERSION 3.24)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
message(STATUS CCACHE_DIR: $ENV{CCACHE_DIR})
message(STATUS "[⚪️ HOST   ⚪️] ${CMAKE_HOST_SYSTEM_NAME}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefiles." FORCE)

# ------------------------------- CUDA Options --------------------------------
set(MFEM_USE_CUDA OFF CACHE BOOL "Enable CUDA" FORCE)
if (${MFEM_USE_CUDA} EQUAL ON)
    message(STATUS "[🟢 CUDA   🟢] Enabled")
    set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)
    # set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "" FORCE)
    # set(CUDA_PATH /usr/tce/packages/cuda/cuda-13.1.1 CACHE PATH "" FORCE)
    # set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc" CACHE PATH "" FORCE)
endif()

# ---------------------------- NVTX Debug File --------------------------------
message(STATUS "[🟢 NVTX   🟢] /usr/WS1/camier1/matrix/usr/src/nvtx")
add_compile_definitions(NVTX_DBG_HPP="/usr/WS1/camier1/matrix/usr/src/nvtx/nvtx_dbg.hpp")
add_compile_definitions(NVTX_DBG_FMT="/usr/WS1/camier1/matrix/usr/src/nvtx/nvtx_fmt.hpp")

# --------------------------------- CXX Host ----------------------------------
# With matrixgcc:
set(CMAKE_C_COMPILER gcc CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER g++ CACHE STRING "" FORCE)
execute_process(COMMAND zsh "-c" "${CMAKE_CXX_COMPILER} --version" 
                OUTPUT_VARIABLE CXX_VERSION      
                OUTPUT_STRIP_TRAILING_WHITESPACE 
                COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "[🟡 CXX    🟡] ${CXX_VERSION}")
if(CXX_VERSION MATCHES "GCC")
    message(STATUS "[🟡 CXX    🟡] GNU-based")
elseif(CXX_VERSION MATCHES "clang")
    message(STATUS "[🟡 CXX    🟡] LLVM-based")
else()
    message(FATAL_ERROR "[❌ CXX ❌] Unknown: ${CXX_VERSION}")
endif()

# ---------------------------------- CLANGD ----------------------------------
message(STATUS "[🟣 clangd 🟣] XDG_CACHE_HOME: $ENV{XDG_CACHE_HOME}")
if(NOT DEFINED "$ENV{XDG_CACHE_HOME}" OR "$ENV{XDG_CACHE_HOME}" STREQUAL "")
else()
    message(FATAL_ERROR "XDG_CACHE_HOME is either not set or empty")
endif()

# ---------------------------------- CCACHE ----------------------------------
execute_process(COMMAND zsh "-c" "${CMAKE_CXX_COMPILER_LAUNCHER} -k cache_dir"
                OUTPUT_VARIABLE CCACHE_DIR 
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "[🟠 ccache 🟠] CCACHE_DIR: ${CCACHE_DIR}")
if(NOT DEFINED CCACHE_DIR OR "${CCACHE_DIR}" STREQUAL "")
    message(FATAL_ERROR "CCACHE_DIR is either not set or empty")
endif()

# --------------------------------- CXX Flags ---------------------------------
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -O1" CACHE STRING "" FORCE)

if (${MFEM_USE_CUDA} EQUAL ON)
    # [FMT] constexpr constructor calls non-constexpr function 
    add_compile_options(-diag-suppress=2417)

    # [FMT] loop is not reachable
    add_compile_options(-diag-suppress=128)

    # warning #186-D: pointless comparison of unsigned integer with zero
    add_compile_options(-diag-suppress=186)

    # annotation is ignored on a non-virtual function("tensor")
    # that is explicitly defaulted on its first declaration
    # add_compile_options(-diag-suppress 20012)

    # virtual missing
    add_compile_options(-diag-suppress=611)

    # calling a __host__ function from a __host__ __device__ function is not allowed
    add_compile_options(-diag-suppress=20011)
    add_compile_options(-diag-suppress=20014)
endif()

# add_compile_options(-Wall)

# add_compile_options(-Xptxas -O0)
# add_compile_options(-Xcompiler -fno-inline)

# add_compile_options(-Wno-unused-result)
# add_compile_options(-Wno-unused-variable)
# add_compile_options(-Wno-c++20-extensions)
# add_compile_options(-Wno-overloaded-virtual)
# add_compile_options(-Wno-deprecated-declarations)

# add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
# add_link_options(-static-libstdc++)

# ---------------------------- MFEM Settings -----------------------------
set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(ENABLE_DOCS OFF)
set(ENABLE_DOXYGEN OFF)
set(ENABLE_EXAMPLES OFF)
set(MFEM_PRECISION "double" CACHE STRING "" FORCE)

option(MFEM_ENABLE_TESTING "Enable the ctest" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" OFF)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)

# ---------------------------- MFEM Options ------------------------------
option(MFEM_USE_MPI "Enable MPI" ON)
set(MPI_PATH "/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-10.3.1-magic" CACHE PATH "" FORCE)
set(MPICXX "${MPI_PATH}/bin/mpicxx" CACHE PATH "" FORCE)

# -------------------------------- HYPRE -----------------------------------
if (${MFEM_USE_CUDA} EQUAL ON)
    set(HYPRE_DIR "/usr/WS1/camier1/matrix/usr/local/hypre-gcc-cuda" CACHE PATH "")
else()
    set(HYPRE_DIR "/usr/WS1/camier1/matrix/usr/local/hypre-gcc-cpu" CACHE PATH "")
endif()

# -------------------------------- METIS -----------------------------------
option(MFEM_USE_METIS "Enable METIS usage" ${MFEM_USE_MPI})
set(METIS_DIR "$ENV{HOME}/home/matrix/usr/local/metis" CACHE PATH "")

# -------------------------------- GSLIB -----------------------------------
option(MFEM_USE_GSLIB "Enable GSLIB" OFF)
set(GSLIB_DIR "$ENV{HOME}/home/matrix/usr/local/gslib" CACHE PATH "")

# ---------------------------- Link OPTIONS --------------------------------
option(MFEM_USE_LIBUNWIND "Enable backtrace for errors." OFF)
option(BUILD_SHARED_LIBS "Enable shared library build of MFEM" OFF)

option(MFEM_ENABLE_TESTING "Enable the ctest" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" OFF)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)
option(MFEM_ENABLE_BENCHMARKS "Build all of the benchmarks" OFF)

# ------------------------- BENCHMARK Options ----------------------------
option(MFEM_USE_BENCHMARK "Enable Benchmarks" ON)
option(MFEM_ENABLE_BENCHMARKS "Build all of the benchmarks" OFF)
set(BENCHMARK_DIR $ENV{HOME}/home/matrix/usr/local/benchmark CACHE PATH "" FORCE)

# ------------------------- FMT Options ----------------------------
set(fmt_DIR "/usr/WS1/camier1/matrix/usr/local/fmt/lib64/cmake/fmt")
find_package(fmt REQUIRED)