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
#   - HYPRE_FOUND
#   - HYPRE_LIBRARIES
#   - HYPRE_INCLUDE_DIRS
#   - HYPRE_VERSION
#   - HYPRE_USING_CUDA (internal)
#   - HYPRE_USING_HIP (internal)

if (HYPRE_FOUND)
  if (HYPRE_USING_CUDA)
    find_package(CUDAToolkit REQUIRED)
  endif()
  if (HYPRE_USING_HIP)
    find_package(rocsparse REQUIRED)
    find_package(rocrand REQUIRED)
  endif()
  if (HYPRE_LIBRARIES AND HYPRE_INCLUDE_DIRS AND HYPRE_VERSION)
    find_package_handle_standard_args(HYPRE
      REQUIRED_VARS HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS HYPRE_VERSION
    )
    return()
  endif()
endif()

include(MfemCmakeUtilities)
mfem_find_package(HYPRE HYPRE HYPRE_DIR "include" "HYPRE.h" "lib" "HYPRE"
  "Paths to headers required by HYPRE." "Libraries required by HYPRE."
  CHECK_BUILD HYPRE_USING_CUDA FALSE
  "
#undef HYPRE_USING_CUDA
#include <HYPRE_config.h>

#ifndef HYPRE_USING_CUDA
#error HYPRE is built without CUDA.
#endif

int main()
{
   return 0;
}
"
  CHECK_BUILD HYPRE_USING_HIP FALSE
  "
#undef HYPRE_USING_HIP
#include <HYPRE_config.h>

#ifndef HYPRE_USING_HIP
#error HYPRE is built without HIP.
#endif

int main()
{
   return 0;
}
")

if (HYPRE_FOUND AND (NOT HYPRE_VERSION))
  try_run(HYPRE_VERSION_RUN_RESULT HYPRE_VERSION_COMPILE_RESULT
          ${CMAKE_CURRENT_BINARY_DIR}/config
          ${CMAKE_CURRENT_SOURCE_DIR}/config/get_hypre_version.cpp
          CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${HYPRE_INCLUDE_DIRS}
          RUN_OUTPUT_VARIABLE HYPRE_VERSION_OUTPUT)
  if ((HYPRE_VERSION_RUN_RESULT EQUAL 0) AND HYPRE_VERSION_OUTPUT)
    string(STRIP "${HYPRE_VERSION_OUTPUT}" HYPRE_VERSION)
    set(HYPRE_VERSION ${HYPRE_VERSION} CACHE STRING "HYPRE version." FORCE)
    message(STATUS "Found HYPRE version ${HYPRE_VERSION}")
  else()
    message(FATAL_ERROR "Unable to determine HYPRE version.")
  endif()
endif()

if (HYPRE_FOUND AND HYPRE_USING_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Initialize CUSPARSE_LIBRARIES, CURAND_LIBRARIES, and CUBLAS_LIBRARIES:
  mfem_culib_set_libraries(CUSPARSE cusparse)
  mfem_culib_set_libraries(CURAND curand)
  mfem_culib_set_libraries(CUBLAS cublas)
  mfem_culib_set_libraries(CUSOLVER cusolver)
  list(APPEND HYPRE_LIBRARIES ${CUSPARSE_LIBRARIES} ${CURAND_LIBRARIES}
       ${CUBLAS_LIBRARIES} ${CUSOLVER_LIBRARIES})
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARIES} CACHE STRING
      "HYPRE libraries + dependencies." FORCE)
  message(STATUS "Updated HYPRE_LIBRARIES: ${HYPRE_LIBRARIES}")
endif()

if (HYPRE_FOUND AND HYPRE_USING_HIP)
  find_package(rocsparse REQUIRED)
  find_package(rocrand REQUIRED)
  list(APPEND HYPRE_LIBRARIES ${rocsparse_LIBRARIES} ${rocrand_LIBRARIES})
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARIES} CACHE STRING
      "HYPRE libraries + dependencies." FORCE)
  message(STATUS "Updated HYPRE_LIBRARIES: ${HYPRE_LIBRARIES}")
endif()

find_package_handle_standard_args(HYPRE
  REQUIRED_VARS HYPRE_LIBRARIES HYPRE_INCLUDE_DIRS HYPRE_VERSION
)
