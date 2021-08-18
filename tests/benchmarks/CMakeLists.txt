# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
    target_link_libraries(bench_${name} mfem pthread)
    add_dependencies(${MFEM_ALL_BENCHMARKS_TARGET_NAME} bench_${name})

    add_test(NAME bench_${name}_cpu
             COMMAND bench_${name} --benchmark_context=device=cpu)

    if (MFEM_USE_CUDA)
        add_test(NAME bench_${name}_cuda
                 COMMAND bench_${name} --benchmark_context=device=cuda)
    endif(MFEM_USE_CUDA)
endfunction(add_benchmark)

#-------------------------------------------------------------------------------
if (MFEM_USE_BENCHMARK)
    add_benchmark(kernels)
    add_benchmark(tmop)
    add_benchmark(vector)
    add_benchmark(virtuals)
endif(MFEM_USE_BENCHMARK)

