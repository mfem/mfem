# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

set(HIOP_EXAMPLES_SRCS)
list(APPEND HIOP_EXAMPLES_SRCS ex9.cpp)

if (MFEM_USE_MPI)
  list(APPEND HIOP_EXAMPLES_SRCS ex9p.cpp)
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_hiop" target, see below.
add_custom_target(test_hiop
  ${CMAKE_CTEST_COMMAND} -R hiop USES_TERMINAL)

# Add one executable per cpp file, adding "hiop_" as prefix. Sets
# "test_hiop" as a target that depends on the given examples.
set(PFX hiop_)
add_mfem_examples(HIOP_EXAMPLES_SRCS ${PFX} "" test_hiop)

# Testing.
# The HIOP tests can be run separately using the target "test_hiop"
# which builds the examples and runs:
#   ctest -R hiop

# Command line options for the tests.
# Example 9:
set(EX9_COMMON_OPTS -m ../../data/periodic-segment.mesh -p 0 -dt 0.005)
set(EX9_TEST_OPTS ${EX9_COMMON_OPTS}  -r  2 )
set(EX9P_TEST_OPTS ${EX9_COMMON_OPTS})

# Add the tests: one test per source file.
foreach(SRC_FILE ${HIOP_EXAMPLES_SRCS})
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
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
      ${MPIEXEC_POSTFLAGS})
  endif()
endforeach()
