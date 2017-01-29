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

# Sets the following variables
#   - ParMETIS_FOUND
#   - ParMETIS_INCLUDE_DIRS
#   - ParMETIS_LIBRARIES
#
# We need the following libraries:
#   parmetis

include(MfemCmakeUtilities)
mfem_find_package(ParMETIS ParMETIS ParMETIS_DIR "include" parmetis.h
  "lib" parmetis
  "Paths to headers required by ParMETIS." "Libraries required by ParMETIS.")
