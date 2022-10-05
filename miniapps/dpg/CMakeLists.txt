# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND DPG_SOURCES
util/weakform.cpp
util/complexweakform.cpp
util/blockstaticcond.cpp
util/complexstaticcond.cpp
util/pml.cpp)

list(APPEND DPG_HEADERS
util/weakform.hpp
util/complexweakform.hpp
util/blockstaticcond.hpp
util/complexstaticcond.hpp
util/pml.hpp)

if (MFEM_USE_MPI)
  list(APPEND DPG_SOURCES
  util/pweakform.cpp
  util/pcomplexweakform.cpp)
  list(APPEND DPG_HEADERS
  util/pweakform.hpp
  util/pcomplexweakform.hpp)
endif ()

convert_filenames_to_full_paths(DPG_SOURCES)
convert_filenames_to_full_paths(DPG_HEADERS)

set(DPG_COMMON_FILES
    EXTRA_SOURCES ${HOOKE_SOURCES}
    EXTRA_HEADERS ${HOOKE_HEADERS})

add_mfem_miniapp(diffusion
  MAIN diffusion.cpp
  ${DPG_COMMON_FILES}
  LIBRARIES mfem)

add_mfem_miniapp(convection-diffusion
MAIN convection-diffusion.cpp
${DPG_COMMON_FILES}
LIBRARIES mfem)

add_mfem_miniapp(acoustics
MAIN acoustics.cpp
${DPG_COMMON_FILES}
LIBRARIES mfem)

add_mfem_miniapp(maxwell
MAIN maxwell.cpp
${DPG_COMMON_FILES}
LIBRARIES mfem)

if (MFEM_USE_MPI)
  add_mfem_miniapp(pdiffusion
    MAIN pdiffusion.cpp
    ${DPG_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(pconvection-diffusion
  MAIN pconvection-diffusion.cpp
  ${DPG_COMMON_FILES}
  LIBRARIES mfem)

  add_mfem_miniapp(pacoustics
  MAIN pacoustics.cpp
  ${DPG_COMMON_FILES}
  LIBRARIES mfem)

  add_mfem_miniapp(pmaxwell
  MAIN pmaxwell.cpp
  ${DPG_COMMON_FILES}
  LIBRARIES mfem)
endif ()

