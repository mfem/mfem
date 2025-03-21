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

set(DFEM_EXAMPLES_SRCS)

if (MFEM_USE_MPI)
    list(APPEND DFEM_EXAMPLES_SRCS minimal_surface.cpp)
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_dfem" target, see below.
add_custom_target(test_dfem
  ${CMAKE_CTEST_COMMAND} -R dfem USES_TERMINAL)

# Add one executable per cpp file, adding "dfem_" as prefix. Sets
# "test_dfem" as a target that depends on the given examples.
set(PFX dfem_)
add_mfem_examples(DFEM_EXAMPLES_SRCS ${PFX} "" test_dfem)

# Testing.
# The DFEM tests can be run separately using the target "test_dfem"
# which builds the examples and runs:
#   ctest -R dfem

if (MFEM_ENABLE_TESTING)
    set(MINIMAL_SURFACE_COMMON_OPTS -o 1 -r 2)
    set(MINIMAL_SURFACE_TEST_OPTS ${MINIMAL_SURFACE_COMMON_OPTS})
    set(MINIMAL_SURFACEP_TEST_OPTS ${MINIMAL_SURFACE_COMMON_OPTS})

    # Add the tests: one test per source file.
    foreach(SRC_FILE ${DFEM_EXAMPLES_SRCS})
        get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
        string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})
        string(TOUPPER ${TEST_NAME} UP_TEST_NAME)
        set(TEST_NAME ${PFX}${TEST_NAME})

        set(THIS_TEST_OPTIONS "-no-vis")
        list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_TEST_OPTS})
        # message(STATUS "Test ${TEST_NAME} options: ${THIS_TEST_OPTIONS}")

        add_test(NAME ${TEST_NAME}_np=4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endforeach()
endif()
