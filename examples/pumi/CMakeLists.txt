# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

set(PUMI_EXAMPLES_SRCS)
# All PUMI examples require MPI
if (MFEM_USE_MPI)
  list(APPEND PUMI_EXAMPLES_SRCS
    ex1.cpp
    ex1p.cpp
    ex2.cpp
    ex6p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_pumi" target, see below.
add_custom_target(test_pumi
  ${CMAKE_CTEST_COMMAND} -R pumi USES_TERMINAL)

# Add one executable per cpp file, adding "pumi_" as prefix. Sets
# "test_pumi" as a target that depends on the given examples.
set(PFX pumi_)
add_mfem_examples(PUMI_EXAMPLES_SRCS ${PFX} "" test_pumi)

# Testing.
# The PUMI tests can be run separately using the target "test_pumi"
# which builds the examples and runs:
#   ctest -R pumi

if (MFEM_ENABLE_TESTING)
  # Command line options for the tests.
  # TODO...

  # Set the number of processors for the parallel examples. The value of
  # MFEM_MPI_NP is ignored.
  set(EX1_TEST_NP 1)
  set(EX1P_TEST_NP 8)
  set(EX2_TEST_NP 1)
  set(EX6P_TEST_NP 8)

  # Add the tests: one test per source file.
  foreach(SRC_FILE ${PUMI_EXAMPLES_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})
    string(TOUPPER ${TEST_NAME} UP_TEST_NAME)
    set(TEST_NAME ${PFX}${TEST_NAME})

    set(THIS_TEST_OPTIONS "-no-vis")
    list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_TEST_OPTS})
    # message(STATUS "Test ${TEST_NAME} options: ${THIS_TEST_OPTIONS}")

    # All PUMI examples require MPI
    if (FALSE)
      add_test(NAME ${TEST_NAME}_ser
        COMMAND ${TEST_NAME} ${THIS_TEST_OPTIONS})
    else()
      set(TEST_NP ${${UP_TEST_NAME}_TEST_NP})
      add_test(NAME ${TEST_NAME}_np=${TEST_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endif()
  endforeach()
endif()
