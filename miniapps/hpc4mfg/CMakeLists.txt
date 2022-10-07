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
  list(APPEND DIST_COMMON_SOURCES
        hpc4solvers.cpp)
  list(APPEND DIST_COMMON_HEADERS
        hpc4solvers.hpp)
  list(APPEND DIST_COMMON_SOURCES
        NLGLSIntegrator.cpp)
  list(APPEND DIST_COMMON_HEADERS
        NLGLSIntegrator.hpp)
  list(APPEND DIST_COMMON_SOURCES
        advection_diffusion_solver.cpp)
  list(APPEND DIST_COMMON_HEADERS
        advection_diffusion_solver.hpp)
  list(APPEND DIST_COMMON_SOURCES
        ascii.cpp)
  list(APPEND DIST_COMMON_HEADERS
        ascii.hpp)

convert_filenames_to_full_paths(DIST_COMMON_SOURCES)
convert_filenames_to_full_paths(DIST_COMMON_HEADERS)

set(DIST_COMMON_FILES
        EXTRA_SOURCES ${DIST_COMMON_SOURCES}
        EXTRA_HEADERS ${DIST_COMMON_HEADERS})

add_mfem_miniapp(ndarcy
    MAIN ndarcy.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(GLSSolver
    MAIN GLS_Solver.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(AdvDiffSolver
    MAIN AdvDiff_Solver.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem mfem-common)


endif ()
