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
  list(APPEND HPC4_COMMON_SOURCES
      ascii.cpp
      hpc4mat.cpp
      hpc4solvers.cpp
      NLGLSIntegrator.cpp
      advection_diffusion_solver.cpp
      MMA.cpp
      Stokes.cpp)
  list(APPEND HPC4_COMMON_HEADERS
    ascii.hpp
      hpc4mat.hpp
      hpc4solvers.hpp
      NLGLSIntegrator.hpp
      advection_diffusion_solver.hpp
      MMA.hpp
      Stokes.hpp)
             
convert_filenames_to_full_paths(DIST_COMMON_SOURCES)
convert_filenames_to_full_paths(DIST_COMMON_HEADERS)

set(HPC4_COMMON_FILES
    EXTRA_SOURCES ${HPC4_COMMON_SOURCES}
    EXTRA_HEADERS ${HPC4_COMMON_HEADERS})

add_mfem_miniapp(ndarcy
    MAIN ndarcy.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(ndarcy_opt
    MAIN ndarcy_opt.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(ndarcy_opt_2
    MAIN ndarcy_opt_2.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(ndarcy_opt_3
    MAIN ndarcy_opt_3.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(GLSSolver
    MAIN GLS_Solver.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(AdvDiffSolver
    MAIN AdvDiff_Solver.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Stokes3DMicro
    MAIN Stokes3DMicro.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Stokes2DMicro
    MAIN Stokes2DMicro.cpp
    ${HPC4_COMMON_FILES}
    LIBRARIES mfem mfem-common)


endif ()
