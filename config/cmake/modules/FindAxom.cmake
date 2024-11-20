# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

include(MfemCmakeUtilities)
# Note: components are enabled based on the find_package() parameters.
mfem_find_package(Axom AXOM AXOM_DIR "include" "" "lib" ""
  "Paths to headers required by Axom." "Libraries required by Axom."
  ADD_COMPONENT core "include" axom/core.hpp "lib" axom_core
  ADD_COMPONENT inlet "include" axom/inlet.hpp "lib" axom_inlet
  ADD_COMPONENT klee "include" axom/klee.hpp "lib" axom_klee
  ADD_COMPONENT lumberjack "include" axom/lumberjack.hpp "lib" axom_lumberjack
  ADD_COMPONENT mint "include" axom/mint.hpp "lib" axom_mint
  ADD_COMPONENT multimat "include" axom/multimat.hpp "lib" axom_multimat
  ADD_COMPONENT quest "include" axom/quest.hpp "lib" axom_quest
  ADD_COMPONENT sidre "include" axom/sidre.hpp "lib" axom_sidre
  ADD_COMPONENT slam "include" axom/slam.hpp "lib" axom_slam
  ADD_COMPONENT slic "include" axom/slic.hpp "lib" axom_slic)
