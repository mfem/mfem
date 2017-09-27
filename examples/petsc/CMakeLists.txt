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

set(PETSC_EXAMPLES_SRCS "")
set(PETSC_RC_FILES "")

if (MFEM_USE_MPI)
  list(APPEND PETSC_EXAMPLES_SRCS
    ex1p.cpp
    ex2p.cpp
    ex3p.cpp
    ex4p.cpp
    ex5p.cpp
    ex6p.cpp
    ex9p.cpp
    ex10p.cpp
    )
  list(APPEND PETSC_RC_FILES
    rc_ex1p
    rc_ex2p
    rc_ex3p rc_ex3p_bddc
    rc_ex4p rc_ex4p_bddc
    rc_ex5p_bddc rc_ex5p_fieldsplit
    rc_ex9p_expl rc_ex9p_impl
    rc_ex10p
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add targets to copy rc_* files from the source directory
foreach(RC_FILE ${PETSC_RC_FILES})
  add_custom_command(OUTPUT ${RC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${RC_FILE} ${RC_FILE}
    COMMENT "copy ${RC_FILE}")
endforeach()
add_custom_target(copy_petsc_rc_files DEPENDS ${PETSC_RC_FILES}
  COMMENT "Copying PETSC example rc files ...")

# Add "test_petsc" target, see below.
add_custom_target(test_petsc
  ${CMAKE_CTEST_COMMAND} -R petsc -C PETSC USES_TERMINAL)

# Add one executable per cpp file, adding "petsc_" as prefix. Sets
# "copy_petsc_rc_files" as a prerequisite for the given examples. Also, sets
# "test_petsc" as a target that depends on these examples.
set(PFX petsc_)
add_mfem_examples(PETSC_EXAMPLES_SRCS ${PFX} copy_petsc_rc_files test_petsc)

# Testing.
# We do not want these tests to be run by default, so we add them to their own
# configuration - PETSC. To run the tests defined here use the "test_petsc"
# target which builds the examples and runs:
#   ctest -C PETSC -R petsc

# Command line options for the tests.
set(EX1P_ARGS -m ../../data/amr-quad.mesh --usepetsc --petscopts rc_ex1p)
set(EX2P_ARGS -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex2p)
set(EX3P_ARGS -m ../../data/klein-bottle.mesh
    -o 2 -f 0.1 --usepetsc --petscopts rc_ex3p_bddc --nonoverlapping)
set(EX4P_ARGS -m ../../data/klein-bottle.mesh
    -o 2 --usepetsc --petscopts rc_ex4p_bddc --nonoverlapping)
set(EX5_BDDC_ARGS -m ../../data/star.mesh
    --usepetsc --petscopts rc_ex5p_bddc --nonoverlapping)
set(EX5_FSPL_ARGS -m ../../data/beam-tet.mesh
    --usepetsc --petscopts rc_ex5p_fieldsplit)
set(EX5P_ARGS ${EX5_FSPL_ARGS})
set(EX6P_ARGS -m ../../data/amr-quad.mesh --usepetsc)
set(EX9P_ARGS -m ../../data/periodic-hexagon.mesh
    --usepetsc --petscopts rc_ex9p_expl)
set(EX10P_ARGS -m ../../data/beam-quad.mesh
    -tf 30 -s 3 -rs 2 -dt 3 --usepetsc --petscopts rc_ex10p)

# Add the tests: one test per source file.
foreach(SRC_FILE ${PETSC_EXAMPLES_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})
  string(TOUPPER ${TEST_NAME} UP_TEST_NAME)
  set(TEST_NAME ${PFX}${TEST_NAME})

  set(THIS_TEST_OPTIONS "-no-vis")
  list(APPEND THIS_TEST_OPTIONS ${${UP_TEST_NAME}_ARGS})
  # message(STATUS "Test ${TEST_NAME} options: ${THIS_TEST_OPTIONS}")

  if (NOT (${TEST_NAME} MATCHES ".*p$"))
    add_test(NAME ${TEST_NAME}_ser
      COMMAND ${TEST_NAME} ${THIS_TEST_OPTIONS}
      CONFIGURATIONS PETSC)
  else()
    add_test(NAME ${TEST_NAME}_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
      ${MPIEXEC_POSTFLAGS}
      CONFIGURATIONS PETSC)
  endif()
endforeach()
