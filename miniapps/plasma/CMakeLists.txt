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

if (MFEM_USE_MPI)
  list(APPEND PLASMA_COMMON_SOURCES
  cold_plasma_dielectric_coefs.cpp
  cold_plasma_dielectric_eb_solver.cpp
  stix_bcs.cpp
  vis_object.cpp)

  list(APPEND PLASMA_COMMON_HEADERS
  plasma.hpp
  cold_plasma_dielectric_coefs.hpp
  cold_plasma_dielectric_eb_solver.hpp
  stix_bcs.hpp
  vis_object.hpp)

  convert_filenames_to_full_paths(PLASMA_COMMON_SOURCES)
  convert_filenames_to_full_paths(PLASMA_COMMON_HEADERS)

  set(PLASMA_COMMON_FILES
      EXTRA_SOURCES ${PLASMA_COMMON_SOURCES}
      EXTRA_HEADERS ${PLASMA_COMMON_HEADERS})

  add_mfem_miniapp(stix_r1d_eb
    MAIN stix_r1d_eb.cpp
    ${PLASMA_COMMON_FILES}
    LIBRARIES mfem-common)
endif()
