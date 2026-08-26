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
# <cublas_v2.h> (via gpu_api_utils.h -> gpu_wrapper.h). Require MFEM to use CUDA
# too, and add the CUDA toolkit include dirs so the CHECK_BUILD probe below
# (compiled with the plain C++ compiler) can find them.
find_file(SuperLUDist_CONFIG_HEADER superlu_dist_config.h
  HINTS ${SuperLUDist_DIR} PATH_SUFFIXES include SRC SuperLU)
if (SuperLUDist_CONFIG_HEADER)
  file(STRINGS ${SuperLUDist_CONFIG_HEADER} SuperLUDist_HAVE_CUDA
    REGEX "^#define HAVE_CUDA")
  if (SuperLUDist_HAVE_CUDA)
    if (NOT MFEM_USE_CUDA)
      message(FATAL_ERROR
        "SuperLU_DIST was built with CUDA support, but MFEM_USE_CUDA is OFF. "
        "Enable MFEM_USE_CUDA or use a SuperLU_DIST build without CUDA.")
    endif()
    list(APPEND SuperLUDist_REQUIRED_PACKAGES "CUDAToolkit")
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
