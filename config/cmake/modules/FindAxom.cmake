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

# Defines the following variables:
#   - AXOM_FOUND
#   - AXOM_LIBRARIES
#   - AXOM_INCLUDE_DIRS
#
# Supports optional/required components using find_package(Axom COMPONENTS ...).
#
# MFEM itself does not depend on Axom, however Tribol does. This module exists
# to support MFEM's Tribol integration (e.g. the contact miniapp).

include(MfemCmakeUtilities)

# Axom is typically provided as an install prefix (AXOM_DIR) for MFEM's make
# build. Prefer that as the hint when searching.
if (NOT AXOM_DIR AND DEFINED ENV{AXOM_DIR} AND NOT "$ENV{AXOM_DIR}" STREQUAL "")
  set(AXOM_DIR "$ENV{AXOM_DIR}")
endif()

# Note: components are enabled based on the find_package() parameters.
mfem_find_package(Axom AXOM AXOM_DIR "include" axom/config.hpp "lib" axom_core
  "Paths to headers required by Axom." "Libraries required by Axom."
  ADD_COMPONENT core
    "include" axom/config.hpp "lib" axom_core
  ADD_COMPONENT slic
    "include" axom/slic.hpp "lib" axom_slic
  ADD_COMPONENT slam
    "include" axom/slam.hpp "lib" axom_slam
  ADD_COMPONENT mint
    "include" axom/mint.hpp "lib" axom_mint
  ADD_COMPONENT primal
    "include" axom/primal.hpp "lib" ""
  ADD_COMPONENT quest
    "include" axom/quest.hpp "lib" axom_quest
  ADD_COMPONENT lumberjack
    "include" axom/lumberjack.hpp "lib" axom_lumberjack
)
