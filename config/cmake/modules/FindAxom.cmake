# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
  ADD_COMPONENT Axom "include" axom/config.hpp "lib" axom)
