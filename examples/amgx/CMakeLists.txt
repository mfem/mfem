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

set(AMGX_EXAMPLES_SRCS)
list(APPEND AMGX_EXAMPLES_SRCS
  ex1.cpp
  )

if (MFEM_USE_MPI)
  list(APPEND AMGX_EXAMPLES_SRCS
    ex1p.cpp
    )
endif()

set(AMGX_JSON_FILES amg_pcg.json multi_gs.json precon.json)

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add targets to copy *.json files from the source directory
foreach(JSON_FILE ${AMGX_JSON_FILES})
  add_custom_command(OUTPUT ${JSON_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${JSON_FILE} ${JSON_FILE}
    COMMENT "copy ${JSON_FILE}")
endforeach()
add_custom_target(copy_amgx_json_files DEPENDS ${AMGX_JSON_FILES}
  COMMENT "Copying AMGX example json files ...")

# Add "test_amgx" target, see below.
add_custom_target(test_amgx
  ${CMAKE_CTEST_COMMAND} -R amgx USES_TERMINAL)

# Add one executable per cpp file, adding "amgx_" as prefix. Sets
# "copy_amgx_json_files" as a prerequisite for the given examples. Also, sets
# "test_amgx" as a target that depends on the given examples.
set(PFX amgx_)
add_mfem_examples(AMGX_EXAMPLES_SRCS ${PFX} copy_amgx_json_files test_amgx)

# Testing.
# The AMGX tests can be run separately using the target "test_amgx"
# which builds the examples and runs:
#   ctest -R amgx

# Command line options for the tests.
# Example 1/1p:
set(EX1_TEST_OPTS)
set(EX1P_TEST_OPTS)

# Add the tests: one test per source file.
foreach(SRC_FILE ${AMGX_EXAMPLES_SRCS})
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
    add_test(NAME ${TEST_NAME}_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
      ${MPIEXEC_POSTFLAGS})
  endif()
endforeach()
