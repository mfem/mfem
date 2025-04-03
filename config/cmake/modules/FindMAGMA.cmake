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
#   - MAGMA_FOUND
#   - MAGMA_LIBRARIES
#   - MAGMA_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(MAGMA MAGMA MAGMA_DIR "include" "magma.h" "lib" "magma"
  "Paths to headers required by MAGMA." "Libraries required by MAGMA.")

if (MAGMA_FOUND AND MFEM_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Initialize CUSPARSE_LIBRARIES and CUBLAS_LIBRARIES:
  mfem_culib_set_libraries(CUSPARSE cusparse)
  mfem_culib_set_libraries(CUBLAS cublas)
  list(APPEND MAGMA_LIBRARIES ${CUSPARSE_LIBRARIES} ${CUBLAS_LIBRARIES})
  set(MAGMA_LIBRARIES ${MAGMA_LIBRARIES} CACHE STRING
      "MAGMA libraries + dependencies." FORCE)
  message(STATUS "Updated MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}")
endif()

if (MAGMA_FOUND AND MFEM_USE_HIP)
  find_package(HIPBLAS REQUIRED)
  find_package(HIPSPARSE REQUIRED)
  list(APPEND MAGMA_LIBRARIES ${HIPBLAS_LIBRARIES} ${HIPSPARSE_LIBRARIES})
  set(MAGMA_LIBRARIES ${MAGMA_LIBRARIES} CACHE STRING
      "MAGMA libraries + dependencies." FORCE)
  message(STATUS "Updated MAGMA_LIBRARIES: ${MAGMA_LIBRARIES}")
endif()
