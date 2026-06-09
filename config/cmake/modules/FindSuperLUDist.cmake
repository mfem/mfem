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

# Sets the following variables:
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARIES

include(MfemCmakeUtilities)
# When SuperLU_DIST is built with CUDA, its public header superlu_defs.h pulls in
# <cublas_v2.h> (via gpu_api_utils.h -> gpu_wrapper.h). This is independent of
# MFEM_USE_CUDA, so detect it from superlu_dist_config.h and, if present, add the
# CUDA toolkit include dirs so the CHECK_BUILD probe below (compiled with the
# plain C++ compiler) and any consumer can find them.
find_file(SuperLUDist_CONFIG_HEADER superlu_dist_config.h
  HINTS ${SuperLUDist_DIR} PATH_SUFFIXES include SRC SuperLU)
if (SuperLUDist_CONFIG_HEADER)
  file(STRINGS ${SuperLUDist_CONFIG_HEADER} SuperLUDist_HAVE_CUDA
    REGEX "^#define HAVE_CUDA")
  if (SuperLUDist_HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(SuperLUDist_REQUIRED_PACKAGES "CUDAToolkit")
  endif()
endif()
mfem_find_package(SuperLUDist SuperLUDist SuperLUDist_DIR
  "include;SRC;SuperLU" "superlu_defs.h;slu_ddefs.h"
  "lib;SRC" "superludist;superlu_dist;superlu_dist_4.3" # add NAMES_PER_DIR?
  "Paths to headers required by SuperLU_DIST."
  "Libraries required by SuperLU_DIST."
  CHECK_BUILD SuperLUDist_VERSION_OK TRUE
  "
#include <superlu_defs.h>
int main()
{
  superlu_dist_options_t opts;
  return 0;
}
")
