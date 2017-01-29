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

# Sets the following variables:
#   - SuperLUDist_FOUND
#   - SuperLUDist_INCLUDE_DIRS
#   - SuperLUDist_LIBRARIES

include(MfemCmakeUtilities)
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
