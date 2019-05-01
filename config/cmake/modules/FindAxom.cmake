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

# Defines the following variables:
#   - AXOM_FOUND
#   - AXOM_LIBRARIES
#   - AXOM_INCLUDE_DIRS

include(MfemCmakeUtilities)
# Note: components are enabled based on the find_package() parameters.
mfem_find_package(Axom AXOM AXOM_DIR "include" "" "lib" ""
  "Paths to headers required by Axom." "Libraries required by Axom."
  ADD_COMPONENT Axom "include" axom/config.hpp "lib" axom)
