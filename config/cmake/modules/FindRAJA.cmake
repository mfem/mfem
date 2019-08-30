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
#   - RAJA_FOUND
#   - RAJA_LIBRARIES
#   - RAJA_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(RAJA RAJA RAJA_DIR "include" "RAJA/RAJA.hpp" "lib" "RAJA"
  "Paths to headers required by RAJA." "Libraries required by RAJA.")

if (NOT RAJA_CONFIG_CMAKE)
   set(RAJA_CONFIG_CMAKE "${RAJA_DIR}/share/raja/cmake/raja-config.cmake")
endif()
if (EXISTS "${RAJA_CONFIG_CMAKE}")
   include("${RAJA_CONFIG_CMAKE}")
   if (ENABLE_CUDA AND NOT MFEM_USE_CUDA)
      message(FATAL_ERROR
         "RAJA is built with CUDA: MFEM_USE_CUDA=YES is required")
   endif()
endif()
