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

set(SUNDIALS_EXAMPLES_SRCS)
list(APPEND SUNDIALS_EXAMPLES_SRCS
  ex9.cpp
  ex10.cpp
  ex16.cpp
  )

if (MFEM_USE_MPI)
  list(APPEND SUNDIALS_EXAMPLES_SRCS
    ex9p.cpp
    ex10p.cpp
    ex16p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_sundials" target, see below.
add_custom_target(test_sundials
  ${CMAKE_CTEST_COMMAND} -R sundials USES_TERMINAL)

# Add one executable per cpp file, adding "sundials_" as prefix. Sets
# "test_sundials" as a target that depends on the given examples.
set(PFX sundials_)
add_mfem_examples(SUNDIALS_EXAMPLES_SRCS ${PFX} "" test_sundials)

# Testing.
# The SUNDIALS tests can be run separately using the target "test_sundials"
# which builds the examples and runs:
#   ctest -R sundials

# Command line options for the tests.
# Example 9: test CVODE with CV_ADAMS (non-stiff implicit) time stepping
set(EX9_COMMON_OPTS -m ../../data/periodic-hexagon.mesh -p 0 -s 7)
set(EX9_TEST_OPTS ${EX9_COMMON_OPTS}  -r  2 -dt 0.0018 -vs 25)
set(EX9P_TEST_OPTS ${EX9_COMMON_OPTS} -rp 1 -dt 0.0009 -vs 50)
# Example 10: test CVODE with CV_BDF (stiff implicit) time stepping
set(EX10_COMMON_OPTS -m ../../data/beam-quad.mesh -o 2 -s 5 -dt 0.15 -tf 6 -vs 10)
set(EX10_TEST_OPTS ${EX10_COMMON_OPTS}  -r  2)
set(EX10P_TEST_OPTS ${EX10_COMMON_OPTS} -rp 1)
# Example 16: use the default options

# Add the tests: one test per source file.
foreach(SRC_FILE ${SUNDIALS_EXAMPLES_SRCS})
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
