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

if (MFEM_USE_MPI)
  add_mfem_miniapp(block-solvers-compare
    MAIN block-solvers-compare.cpp
    EXTRA_SOURCES div_free_solver.cpp
    EXTRA_HEADERS div_free_solver.hpp
    LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  add_test(NAME block-solvers-compare-constant
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:block-solvers-compare> -r 2
    ${MPIEXEC_POSTFLAGS})

  add_test(NAME block-solvers-compare-high-contrast
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:block-solvers-compare> -r 2
    -m ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.mesh
    -c ${CMAKE_CURRENT_SOURCE_DIR}/high_contrast_coef.txt
    -eb ${CMAKE_CURRENT_SOURCE_DIR}/ess_bdr_attr.txt
    ${MPIEXEC_POSTFLAGS})
endif()
