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
#   - MAGMA_FOUND
#   - MAGMA_LIBRARIES
#   - MAGMA_INCLUDE_DIRS

include(MfemCmakeUtilities)
set(MAGMA_REQUIRED_LIBRARIES cublas cusparse)
mfem_find_package(MAGMA MAGMA MAGMA_DIR "include" "magma.h" "lib" "magma"
  "Paths to headers required by MAGMA." "Libraries required by MAGMA.")
# Make sure the library location is locked down
foreach(lib ${MAGMA_REQUIRED_LIBRARIES})
  list(APPEND MAGMA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/lib${lib}${CMAKE_SHARED_LIBRARY_SUFFIX})
endforeach()
