# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(reconstruction
  MAIN reconstruction.cpp LIBRARIES mfem)

if (MFEM_USE_MPI)
  add_mfem_miniapp(p-reconstruction
    MAIN p-reconstruction.cpp LIBRARIES mfem)
endif()

if (MFEM_USE_SAMRAI)
  enable_language(Fortran)
  add_mfem_miniapp(samrai-advection-diffusion
    MAIN samrai-advection-diffusion.cpp LIBRARIES mfem ${SAMRAI_LIBRARIES}
    EXTRA_SOURCES MeshOps.cpp samrai/LinAdv.cpp samrai/fortran/flux2d.f
                  samrai/fortran/flux3d.f samrai/fortran/fluxcorner3d.f
                  samrai/fortran/grad2d.f samrai/fortran/grad3d.f
                  samrai/fortran/init2d.f samrai/fortran/init3d.f
                  samrai/fortran/stable2d.f samrai/fortran/stable3d.f
                  samrai/fortran/stufprobc.f samrai/fortran/trace1d.f
                  samrai/fortran/trace2d.f samrai/fortran/trace3d.f)
  target_include_directories(samrai-advection-diffusion PUBLIC ${SAMRAI_INCLUDE_DIRS})
  # Copy SAMRAI input files to build directory
  configure_file(samrai/linadv_input.2d linadv_input.2d COPYONLY)
  configure_file(samrai/linadv_input.3d linadv_input.3d COPYONLY)
endif()
