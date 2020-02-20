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

list(APPEND ALL_EXE_SRCS
  ex1.cpp
  ex2.cpp
  ex3.cpp
  ex4.cpp
  ex5.cpp
  ex6.cpp
  ex7.cpp
  ex8.cpp
  ex9.cpp
  ex10.cpp
  ex14.cpp
  ex15.cpp
  ex16.cpp
  ex17.cpp
  ex18.cpp
  ex19.cpp
  ex20.cpp
  ex21.cpp
  ex22.cpp
  )

if (MFEM_USE_MPI)
  list(APPEND ALL_EXE_SRCS
    ex1p.cpp
    ex2p.cpp
    ex3p.cpp
    ex4p.cpp
    ex5p.cpp
    ex6p.cpp
    ex7p.cpp
    ex8p.cpp
    ex9p.cpp
    ex10p.cpp
    ex11p.cpp
    ex12p.cpp
    ex13p.cpp
    ex14p.cpp
    ex15p.cpp
    ex16p.cpp
    ex17p.cpp
    ex18p.cpp
    ex19p.cpp
    ex20p.cpp
    ex21p.cpp
    ex22p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add one executable per cpp file
add_mfem_examples(ALL_EXE_SRCS)

# Add a test for each example
foreach(SRC_FILE ${ALL_EXE_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TEST_NAME ${SRC_FILENAME})

  set(THIS_TEST_OPTIONS "-no-vis")
  if (${TEST_NAME} MATCHES "ex10p*")
    list(APPEND THIS_TEST_OPTIONS "-tf" "5")
  elseif(${TEST_NAME} MATCHES "ex15p*")
    list(APPEND THIS_TEST_OPTIONS "-e" "1")
  endif()

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

# If STRUMPACK is enabled, add a test run that uses it.
if (MFEM_USE_STRUMPACK)
  add_test(NAME ex11p_strumpack_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:ex11p> "-no-vis" "--strumpack"
    ${MPIEXEC_POSTFLAGS})
endif()

# Include the examples/sundials directory if SUNDIALS is enabled.
if (MFEM_USE_SUNDIALS)
  add_subdirectory(sundials)
endif()

# Include the examples/sundials directory if SUNDIALS is enabled.
if (MFEM_USE_GINKGO)
  add_subdirectory(ginkgo)
endif()

# Include the examples/petsc directory if PETSc is enabled.
if (MFEM_USE_PETSC)
  add_subdirectory(petsc)
endif()

# Include the examples/pumi directory if PUMI is enabled
if (MFEM_USE_PUMI)
  add_subdirectory(pumi)
endif()

if (MFEM_USE_HIOP)
  add_subdirectory(hiop)
endif()

