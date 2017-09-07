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
#   - Scotch_FOUND
#   - Scotch_INCLUDE_DIRS
#   - Scotch_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(Scotch Scotch Scotch_DIR "" "" "" ""
  "Paths to headers required by Scotch."
  "Libraries required by Scotch."
  ADD_COMPONENT "scotch" "include" scotch.h "lib" scotch
  ADD_COMPONENT "scotcherr" "" "" "lib" scotcherr
  ADD_COMPONENT "scotcherrexit" "" "" "lib" scotcherrexit
  ADD_COMPONENT "scotchmetis" "include" "metis.h" "lib" scotchmetis
  ADD_COMPONENT "ptscotch" "include" ptscotch.h "lib" ptscotch
  ADD_COMPONENT "ptscotcherr" "" "" "lib" ptscotcherr
  ADD_COMPONENT "ptscotcherrexit" "" "" "lib" ptscotcherrexit
  ADD_COMPONENT "ptscotchparmetis" "include" "parmetis.h" "lib" ptscotchparmetis
  )
