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
