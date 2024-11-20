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

# Add one executable per cpp file, adding "sundials_" as prefix so the CMake
# target is unique from those in the non-SUNDIALS examples. Also sets
# "test_sundials" as a target that depends on the given SUNDIALS examples.
set(PFX sundials_)
add_mfem_examples(SUNDIALS_EXAMPLES_SRCS ${PFX} "" test_sundials)

# Remove "sundials_" prefix from exectuable name for consistency with GNU build
# system.
foreach(SRC_FILE ${SUNDIALS_EXAMPLES_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TARGET_NAME "${PFX}${SRC_FILENAME}")
  string(REPLACE ${PFX} "" EXE_NAME ${TARGET_NAME})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXE_NAME})
endforeach()

# Testing.
# The SUNDIALS tests can be run separately using the target "test_sundials"
# which builds the examples and runs:
#   ctest -R sundials

if (MFEM_ENABLE_TESTING)
  # Command line options for the tests.
  # Example 9: test CVODE with CV_ADAMS (non-stiff implicit) time stepping
  set(EX9_COMMON_OPTS -m ../../data/periodic-hexagon.mesh -p 0 -s 7)
  set(EX9_TEST_OPTS ${EX9_COMMON_OPTS}  -r  2 -dt 0.0018 -vs 25)
  set(EX9P_TEST_OPTS ${EX9_COMMON_OPTS} -rp 1 -dt 0.0009 -vs 50)
  # Example 10: test CVODE with CV_BDF (stiff implicit) time stepping
  set(EX10_COMMON_OPTS -m ../../data/beam-quad.mesh -o 2 -s 5 -dt 0.15 -tf 6 -vs 10)
  set(EX10_TEST_OPTS ${EX10_COMMON_OPTS}  -r  2)
  set(EX10P_TEST_OPTS ${EX10_COMMON_OPTS} -rp 1)
  # Example 16: test ARKODE with implicit time stepping using mass form
  set(EX16_COMMON_OPTS -s 15)
  set(EX16_TEST_OPTS ${EX16_COMMON_OPTS})
  set(EX16P_TEST_OPTS ${EX16_COMMON_OPTS})

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
      add_test(NAME ${TEST_NAME}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endif()
  endforeach()

  # Add CUDA/HIP tests.
  set(DEVICE_EXAMPLES
      # serial examples with device support:
      ex9
      # parallel examples with device support:
      ex9p)
  set(MFEM_TEST_DEVICE)
  if (MFEM_USE_CUDA)
    set(MFEM_TEST_DEVICE "cuda")
  elseif (MFEM_USE_HIP)
    set(MFEM_TEST_DEVICE "hip")
  endif()
  if (MFEM_TEST_DEVICE)
    foreach(TEST_NAME ${DEVICE_EXAMPLES})
      string(TOUPPER ${TEST_NAME} UP_TEST_NAME)

      set(THIS_TEST_OPTIONS "-no-vis" "-d" "${MFEM_TEST_DEVICE}")
      list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_TEST_OPTS})

      if (NOT (${TEST_NAME} MATCHES ".*p$"))
        add_test(NAME ${PFX}${TEST_NAME}_${MFEM_TEST_DEVICE}_ser
          COMMAND ${PFX}${TEST_NAME} ${THIS_TEST_OPTIONS})
      else()
        add_test(NAME ${PFX}${TEST_NAME}_${MFEM_TEST_DEVICE}_np=${MFEM_MPI_NP}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:${PFX}${TEST_NAME}> ${THIS_TEST_OPTIONS}
          ${MPIEXEC_POSTFLAGS})
      endif()
    endforeach()
  endif(MFEM_TEST_DEVICE)
endif(MFEM_ENABLE_TESTING)
