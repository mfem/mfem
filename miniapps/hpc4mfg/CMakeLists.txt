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
      hpc4DiffMat.cpp
      hpc4solvers.cpp
      NLGLSIntegrator.cpp
      advection_diffusion_solver.cpp
      MMA.cpp
      Stokes.cpp
      ThermalComplianceQoI.cpp)
  list(APPEND HPC4_COMMON_HEADERS
    ascii.hpp
      hpc4mat.hpp
      hpc4DiffMat.hpp
      hpc4solvers.hpp
      NLGLSIntegrator.hpp
      advection_diffusion_solver.hpp
      MMA.hpp
      Stokes.hpp
      ThermalComplianceQoI.hpp)
             
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

add_mfem_miniapp(Full3D
    MAIN Full3D.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D
    MAIN Full2D.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D_Darcy
    MAIN Full2D_Darcy.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D_channel
    MAIN Full2D_channel.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D_channel_periodic
    MAIN Full2D_channel_periodic.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D_channel_periodic_labrynth
    MAIN Full2D_channel_periodic_labrynth.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full3D_channel_periodic_gyroid
    MAIN Full3D_channel_periodic_gyroid.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(Full2D_channel_periodic_tri
    MAIN Full2D_channel_periodic_tri.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(HeatExchangerOpt
    MAIN HeatExchangerOpt.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(HeatExchangerOpt_2
    MAIN HeatExchangerOpt_2.cpp
    ${HPC4_COMMON_FILES} ${NAVIER_COMMON_FILES}
    LIBRARIES mfem mfem-common)
   

endif ()
