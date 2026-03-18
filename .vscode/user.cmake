# m distclean && m serial CXX="ccache clang++"

# m distclean && m serial MFEM_USE_GLVIS=YES CXX="ccache clang++"

# -------------------------------- CMake Setup --------------------------------
cmake_minimum_required(VERSION 3.11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
message(STATUS "CCACHE_DIR: $ENV{CCACHE_DIR}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefiles." FORCE)

# ---------------------------- NVTX Debug File --------------------------------
add_compile_definitions(NVTX_DBG_HPP="$ENV{HOME}/home/mfem/stash/debug/nvtx_dbg.hpp")
add_compile_definitions(NVTX_FMT_HPP="$ENV{HOME}/home/mfem/stash/debug/nvtx_fmt.hpp")

# ------------------------------- HIP Options --------------------------------
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    set(MFEM_USE_HIP OFF CACHE BOOL "Enable HIP" FORCE)
    if (MFEM_USE_HIP)
        set(CMAKE_C_COMPILER /usr/lib64/ccache/clang)
        set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
        set(ROCM_PATH /opt/rocm-6.3.1 CACHE PATH "" FORCE)
        set(HIP_ARCH gfx942 CACHE STRING "Target HIP architecture" FORCE)
        # add_compile_options(-funroll-loops)
        # add_compile_options(--offload-arch=gfx942)
        # add_compile_options(-Rpass-analysis=kernel-resource-usage)
        # add_link_options(-lamd_smi)
    endif(MFEM_USE_HIP)
endif()

# --------------------------------- CXX Setup ---------------------------------
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    # set(CC /opt/homebrew/bin/gcc-15)
    # set(CXX /opt/homebrew/bin/g++-15)
    # set(CXX /opt/homebrew/bin/mpicxx)
    # set(CXX /opt/homebrew/opt/llvm/bin/clang++)
    add_compile_options(-I/opt/homebrew/opt/llvm/include/c++/v1)
    # add_compile_options(-I/opt/homebrew/opt/boost/include)
    add_compile_options(-nostdinc++)
    add_link_options(-L/opt/homebrew/opt/llvm/lib/c++)
    set(CXX /usr/bin/clang++)
else()
   message(FATAL_ERROR "Unsupported system")
endif()

execute_process(COMMAND zsh "-c" "${CXX} --version" OUTPUT_VARIABLE CXX_VERSION      
                OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(CXX_VERSION MATCHES "GCC")
    message(STATUS "[🟡 CXX    🟡] GNU-based")
elseif(CXX_VERSION MATCHES "clang")
    message(STATUS "[🟡 CXX    🟡] LLVM-based")
else()
    message(FATAL_ERROR "[❌ CXX ❌] Unknown: ${CXX_VERSION}")
endif()

# --------------------------------- CXX Flags ---------------------------------
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2" CACHE STRING "" FORCE)

add_compile_options(-Wall)
add_compile_options(-pedantic)
# add_compile_options(-Werror)
# add_compile_options(-Wshadow)
# add_compile_options(-pedantic-errors)
add_compile_options(-Wuninitialized)
add_compile_options(-Wfloat-conversion)
add_compile_options(-Wno-c++20-extensions)

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    add_compile_options(-Wno-unused-result)
    add_compile_options(-Wno-deprecated-declarations)
elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
endif()

if(CXX_VERSION MATCHES "clang")
    add_compile_options(-march=native -mtune=native)
    add_compile_options(-Wno-unused-variable)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-nan-infinity-disabled)
    add_compile_options(-Wno-cast-function-type-mismatch)
    add_compile_options(-Wno-mathematical-notation-identifier-extension)
elseif(CXX_VERSION MATCHES "GCC")
    add_compile_options(-Wno-maybe-uninitialized)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
else()
    #🔥🔥🔥 UndefinedBehaviorSanitizer
    # add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    # add_link_options(-fsanitize=undefined)
    #🔥🔥🔥 AddressSanitizer
    # add_compile_options(-fsanitize=address
    #                    -fno-omit-frame-pointer
    #                    -fno-optimize-sibling-calls
    #                    -fsanitize-address-use-after-scope)
    # add_link_options(-fsanitize=address)
    #🔥🔥🔥 Maths and Exceptions
    # add_compile_options(-ffast-math)
    # add_compile_options(-fno-math-errno -fno-signed-zeros)
    # add_compile_options(-fno-exceptions)
endif()

# ---------------------------- MFEM Configuration -----------------------------
set(ENABLE_DOCS OFF)
set(ENABLE_DOXYGEN OFF)
set(ENABLE_EXAMPLES OFF)
set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MFEM_PRECISION "double" CACHE STRING "" FORCE)

# ------------------------------ GLVis Options ---------------------------------
# add_link_options(-Wl,-rpath,/opt/homebrew/lib)
set(GLVIS_DIR $ENV{HOME}/home/glvis/glvis-lib)

# ------------------------------ MFEM Options ---------------------------------
option(MFEM_USE_MPI "Enable MPI" ON)
option(MFEM_USE_METIS "Enable METIS usage" ${MFEM_USE_MPI})

option(MFEM_USE_LIBUNWIND "Enable backtrace for errors." OFF)
option(BUILD_SHARED_LIBS "Enable shared library build of MFEM" OFF)

option(MFEM_USE_METAL "Enable METAL" OFF)
option(MFEM_USE_ENZYME "Enable Enzyme" OFF)
option(MFEM_USE_BENCHMARK "Enable Benchmarks" OFF)

option(MFEM_ENABLE_TESTING "Enable the ctest" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" ON)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)
option(MFEM_ENABLE_BENCHMARKS "Build all of the benchmarks" OFF)

# ---------------------------------- CCACHE ----------------------------------
execute_process(COMMAND zsh "-c" "${CMAKE_CXX_COMPILER_LAUNCHER} -k cache_dir"
                OUTPUT_VARIABLE CCACHE_DIR 
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "[🟠 ccache 🟠] CCACHE_DIR: ${CCACHE_DIR}")
if(NOT DEFINED CCACHE_DIR OR "${CCACHE_DIR}" STREQUAL "")
    message(FATAL_ERROR "CCACHE_DIR is either not set or empty")
endif()

# ---------------------------------- CLANGD ----------------------------------
message(STATUS "[🟣 clangd 🟣] XDG_CACHE_HOME: $ENV{XDG_CACHE_HOME}")
if(NOT DEFINED $ENV{XDG_CACHE_HOME} OR "$ENV{XDG_CACHE_HOME}" STREQUAL "")
else()
    message(FATAL_ERROR "XDG_CACHE_HOME is either not set or empty")
endif()

# ---------------------------------- ENZYME ----------------------------------
if (MFEM_USE_ENZYME)
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        set(ENZYME_DIR "$ENV{HOME}/home/tuo/usr/local/enzyme" CACHE PATH "")
    elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
        set(ENZYME_DIR "/opt/homebrew/opt/enzyme" CACHE PATH "")
    endif()
endif()

# ----------------------------------- HYPRE -----------------------------------
if (MFEM_USE_MPI)
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        set(HYPRE_DIR $ENV{HOME}/home/tuo/usr/local/hypre-hip-aware CACHE PATH "")
    elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
        if (MFEM_PRECISION STREQUAL "double")
            set(HYPRE_DIR "/opt/homebrew/opt/hypre" CACHE PATH "")
        else()
            set(HYPRE_DIR $ENV{HOME}/usr/src/hypre/src/hypre CACHE PATH "")
        endif()
    endif()
endif()

# ----------------------------------- Metis -----------------------------------
if (MFEM_USE_METIS)
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        option(MFEM_USE_METIS_5 ON)
        set(METIS_DIR $ENV{HOME}/home/tuo/usr/local/metis CACHE PATH "")
    elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
        set(METIS_DIR "/opt/homebrew/opt/metis" CACHE PATH "")
    endif()
endif()

# -------------------------------- Benchmarks --------------------------------
if (MFEM_USE_BENCHMARK)
    if(CXX_VERSION MATCHES "clang")
        if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
            set(BENCHMARK_DIR $ENV{HOME}/home/tuo/usr/local/benchmark CACHE PATH "" FORCE)
        elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
            set(BENCHMARK_DIR /opt/homebrew/opt/google-benchmark CACHE PATH "" FORCE)
        endif()
    elseif(CXX_VERSION MATCHES "GCC")
        set(BENCHMARK_DIR /Users/camierjs/usr/local/benchmark-g++-14 CACHE PATH "" FORCE)
    endif()
endif()

# ---------------------------- TRACY Profiler --------------------------------
# include(FetchContent)
# FetchContent_Declare(tracy
#     GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#     GIT_TAG        master
#     GIT_SHALLOW    TRUE
#     GIT_PROGRESS   TRUE)
# FetchContent_MakeAvailable(tracy)
# add_compile_options(-I/Users/camierjs/home/gui/tracy/public/tracy)
# add_compile_options(-fno-omit-frame-pointer)
