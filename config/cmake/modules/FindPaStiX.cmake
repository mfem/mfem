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
#   - PASTIX_FOUND
#   - PASTIX_LIBRARIES
#   - PASTIX_INCLUDE_DIRS

include(MfemCmakeUtilities)
set(PASTIX_REQUIRED_LIBRARIES Threads)
mfem_find_package(PASTIX PASTIX PASTIX_DIR "include" "" "lib" ""
  "Paths to headers required by PaStiX." "Libraries required by PaStiX.")
mfem_find_package(PASTIX_kernels PASTIX_kernels PASTIX_DIR "include" "pastix.h" "lib" "pastix_kernels"
  "Paths to headers required by PaStiX." "Libraries required by PaStiX.")
mfem_find_package(SPM SPM PASTIX_DIR "include" "spm.h" "lib" "spm"
  "Paths to headers required by PaStiX's SPM component." "Libraries required by PaStiX's SPM component.")
find_package(Threads)
list(APPEND PASTIX_LIBRARIES ${SPM_LIBRARIES})
list(APPEND PASTIX_INCLUDE_DIRS ${SPM_INCLUDE_DIRS})
list(APPEND PASTIX_LIBRARIES ${PASTIX_kernels_LIBRARIES})
list(APPEND PASTIX_INCLUDE_DIRS ${PASTIX_kernels_INCLUDE_DIRS})
