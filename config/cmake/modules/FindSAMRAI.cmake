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

# Find SAMRAI.
# Defines the following variables:
#   - SAMRAI_FOUND
#   - SAMRAI_LIBRARIES
#   - SAMRAI_INCLUDE_DIRS

include(MfemCmakeUtilities)

# SAMRAI consists of multiple component libraries
set(SAMRAI_COMPONENTS
  SAMRAI_tbox
  SAMRAI_hier
  SAMRAI_xfer
  SAMRAI_pdat
  SAMRAI_math
  SAMRAI_mesh
  SAMRAI_geom
  SAMRAI_solv
  SAMRAI_algs
  SAMRAI_appu)

mfem_find_package(SAMRAI SAMRAI SAMRAI_DIR
  "include" "SAMRAI/SAMRAI_config.h"
  "lib" "${SAMRAI_COMPONENTS}"
  "Paths to headers required by SAMRAI."
  "Libraries required by SAMRAI.")
