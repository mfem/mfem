# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_MPI)
  add_mfem_miniapp(block-solvers
    MAIN block-solvers.cpp
    EXTRA_SOURCES div_free_solver.cpp
    EXTRA_HEADERS div_free_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(plor
    MAIN plor.cpp
    EXTRA_HEADERS lor_mms.hpp
    LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  add_test(NAME block-solvers-constant
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:block-solvers> -r 2
    ${MPIEXEC_POSTFLAGS})

  add_test(NAME block-solvers-anisotropic
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:block-solvers> -r 2
    -m ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.mesh
    -c ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.coeff
    -eb ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.brd
    ${MPIEXEC_POSTFLAGS})
endif()

add_mfem_miniapp(lor
  MAIN lor.cpp
  EXTRA_HEADERS lor_mms.hpp
  LIBRARIES mfem)
