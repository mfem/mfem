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
include_directories(${PROJECT_BINARY_DIR})

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
set(EX5P_BDDC_ARGS -m ../../data/star.mesh
    --usepetsc --petscopts rc_ex5p_bddc --nonoverlapping)
set(EX5P_FSPL_ARGS -m ../../data/beam-tet.mesh
    --usepetsc --petscopts rc_ex5p_fieldsplit)
set(EX6P_ARGS -m ../../data/amr-quad.mesh --usepetsc)
set(EX9P_E_ARGS -m ../../data/periodic-hexagon.mesh
    --usepetsc --petscopts rc_ex9p_expl -dt 0.1)
set(EX9P_ES_ARGS -m ../../data/periodic-hexagon.mesh
    --usepetsc --petscopts rc_ex9p_expl --no-step)
set(EX9P_IS_ARGS -m ../../data/periodic-hexagon.mesh
    --usepetsc --petscopts rc_ex9p_impl --implicit -tf 0.5)
set(EX10P_ARGS -m ../../data/beam-quad.mesh
    -tf 30 -s 3 -rs 2 -dt 3 --usepetsc --petscopts rc_ex10p)

# Add the tests: one test per command-line-variable.
foreach(TEST_OPTIONS_VAR
        EX1P_ARGS EX2P_ARGS EX3P_ARGS EX4P_ARGS EX5P_BDDC_ARGS EX5P_FSPL_ARGS
        EX6P_ARGS EX9P_E_ARGS EX9P_ES_ARGS EX9P_IS_ARGS EX10P_ARGS)
  string(REGEX REPLACE "^(.+)_ARGS" "\\1" TEST_NAME_UC ${TEST_OPTIONS_VAR})
  string(TOLOWER ${TEST_NAME_UC} TEST_NAME_FULL)
  string(REGEX REPLACE "^([^_]+).*" "\\1" TEST_NAME ${TEST_NAME_FULL})
  set(TEST_NAME_FULL ${PFX}${TEST_NAME_FULL})
  set(TEST_NAME ${PFX}${TEST_NAME})
  set(TEST_OPTIONS "-no-vis" ${${TEST_OPTIONS_VAR}})
  # message(STATUS "${TEST_NAME_FULL} --> ${TEST_NAME} ${TEST_OPTIONS}")

  # All PETSC tests are parallel.
  add_test(NAME ${TEST_NAME_FULL}_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:${TEST_NAME}> ${TEST_OPTIONS}
    ${MPIEXEC_POSTFLAGS}
    CONFIGURATIONS PETSC)
endforeach()
