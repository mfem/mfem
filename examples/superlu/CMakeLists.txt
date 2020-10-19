# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

set(SUPERLU_EXAMPLES_SRCS)
if (MFEM_USE_MPI)
  list(APPEND SUPERLU_EXAMPLES_SRCS
    ex1p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_superlu" target, see below.
add_custom_target(test_superlu
  ${CMAKE_CTEST_COMMAND} -R superlu USES_TERMINAL)

# Add one executable per cpp file, adding "superlu_" as prefix. Sets
# "test_superlu" as a target that depends on the given examples.
set(PFX superlu_)
add_mfem_examples(SUPERLU_EXAMPLES_SRCS ${PFX} "" test_superlu)

# Testing.
# The SuperLU tests can be run separately using the target "test_superlu"
# which builds the examples and runs:
#   ctest -R superlu

# Command line options for the tests.
# Example 1: Test SuperLU on the simple Poisson problem
set(EX1_COMMON_OPTS -m ../../data/star.mesh -p 2)
set(EX1P_TEST_OPTS ${EX1_COMMON_OPTS})

# Add the tests: one test per source file.
foreach(SRC_FILE ${SUPERLU_EXAMPLES_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})
  string(TOUPPER ${TEST_NAME} UP_TEST_NAME)
  set(TEST_NAME ${PFX}${TEST_NAME})

  set(THIS_TEST_OPTIONS "-no-vis")
  list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_TEST_OPTS})
  # message(STATUS "Test ${TEST_NAME} options: ${THIS_TEST_OPTIONS}")

  if (NOT (${TEST_NAME} MATCHES ".*p$"))
    add_test(NAME ${TEST_NAME}_ser
      COMMAND ${TEST_NAME} ${THIS_TEST_OPTIONS})
  else()
    add_test(NAME ${TEST_NAME}_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
      ${MPIEXEC_POSTFLAGS})
  endif()
endforeach()
