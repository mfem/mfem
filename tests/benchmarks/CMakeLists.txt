# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Include the top mfem source directory - needed by some tests, e.g. to
# #include "general/forall.hpp".
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Include the build directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Include the Google Benchmark include directory
include_directories(BEFORE ${BENCHMARK_INCLUDE_DIRS})

# When benchmarking with CodSpeed (MFEM_USE_CODSPEED is set by the user), Google
# Benchmark is replaced by CodSpeed's compatibility layer: it is fetched here and
# the benchmarks are linked against the 'benchmark::benchmark' target, which
# takes care of the include directories and the compile definitions.
set(MFEM_BENCHMARK_EXTRA_LIBS "")
if (MFEM_BENCHMARK_USE_CODSPEED)
   if (CMAKE_VERSION VERSION_LESS 3.18.0)
      message(FATAL_ERROR "Benchmarking with CodSpeed requires CMake >= 3.18")
   endif()
   include(FetchContent)
   set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
   set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
   set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
   set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
   FetchContent_Declare(google_benchmark
      GIT_REPOSITORY https://github.com/CodSpeedHQ/codspeed-cpp
      GIT_TAG v2.4.0
      SOURCE_SUBDIR google_benchmark)
   FetchContent_MakeAvailable(google_benchmark)
   set(MFEM_BENCHMARK_EXTRA_LIBS benchmark::benchmark)
endif()

#-------------------------------------------------------------------------------
# Function to add one benchmark from the tests/benchmarks directory.
#-------------------------------------------------------------------------------
function(add_benchmark name)
    string(TOUPPER ${name} NAME)

    set(${NAME}_BENCH_SRCS bench_${name}.cpp)

    if (MFEM_USE_CUDA)
       set_property(SOURCE ${${NAME}_BENCH_SRCS} PROPERTY LANGUAGE CUDA)
    endif(MFEM_USE_CUDA)

    add_executable(bench_${name} ${${NAME}_BENCH_SRCS})
    target_link_libraries(bench_${name} mfem pthread
                          ${MFEM_BENCHMARK_EXTRA_LIBS})
    add_dependencies(${MFEM_ALL_BENCHMARKS_TARGET_NAME} bench_${name})

    add_test(NAME bench_${name}_cpu
             COMMAND bench_${name} --benchmark_context=device=cpu)

    if (MFEM_USE_CUDA)
        add_test(NAME bench_${name}_cuda
                 COMMAND bench_${name} --benchmark_context=device=cuda)
    endif(MFEM_USE_CUDA)
    if (MFEM_USE_HIP)
        add_test(NAME bench_${name}_hip
                 COMMAND bench_${name} --benchmark_context=device=hip)
    endif(MFEM_USE_HIP)
endfunction(add_benchmark)

#-------------------------------------------------------------------------------
add_benchmark(assembly_levels)
add_benchmark(ceed)
add_benchmark(dg_amr)
add_benchmark(elasticity)
add_benchmark(tmop)
add_benchmark(vector)
add_benchmark(virtuals)
add_benchmark(nlvc)
