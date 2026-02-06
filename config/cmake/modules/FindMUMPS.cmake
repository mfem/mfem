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

# Sets the following variables:
#   - MUMPS_FOUND
#   - MUMPS_LIBRARIES
#   - MUMPS_INCLUDE_DIRS
#   - MUMPS_VERSION

include(MfemCmakeUtilities)

# Toggle which precision of MUMPS to use depending on the precision of MFEM.
if (MFEM_USE_DOUBLE)
  set(_mumps_header dmumps_c.h)
  set(_mumps_lib dmumps)
  set(_cmumps_header zmumps_c.h)
  set(_cmumps_lib zmumps)
elseif(MFEM_USE_SINGLE)
  set(_mumps_header smumps_c.h)
  set(_mumps_lib smumps)
  set(_cmumps_header cmumps_c.h)
  set(_cmumps_lib cmumps)
endif()

mfem_find_package(MUMPS MUMPS MUMPS_DIR
  "include" ${_mumps_header} "lib" ${_mumps_lib}
  "Paths to headers required by MUMPS."
  "Libraries required by MUMPS."
  ADD_COMPONENT mumps_common "include" ${_mumps_header} "lib" mumps_common
  ADD_COMPONENT pord "include" ${_mumps_header} "lib" pord)

if (MFEM_USE_COMPLEX_MUMPS)
  mfem_find_package(COMPLEX_MUMPS MUMPS MUMPS_DIR
    "include" ${_cmumps_header} "lib" ${_cmumps_lib}
    "Paths to headers required by complex MUMPS."
    "Libraries required by complex MUMPS."
    ADD_COMPONENT mumps_common "include" ${_cmumps_header} "lib" mumps_common
    ADD_COMPONENT pord "include" ${_cmumps_header} "lib" pord)

  if (COMPLEX_MUMPS_FOUND)
    # If only complex MUMPS is requested/enabled, make sure MUMPS is considered found.
    set(MUMPS_FOUND TRUE)
    # Append complex include dirs and libs (no deduplication).
    list(APPEND MUMPS_INCLUDE_DIRS ${COMPLEX_MUMPS_INCLUDE_DIRS})
    list(APPEND MUMPS_LIBRARIES    ${COMPLEX_MUMPS_LIBRARIES})
  endif()
endif()

if (MUMPS_FOUND AND (NOT MUMPS_VERSION))
  try_run(MUMPS_VERSION_RUN_RESULT MUMPS_VERSION_COMPILE_RESULT
          ${CMAKE_CURRENT_BINARY_DIR}/config
          ${CMAKE_CURRENT_SOURCE_DIR}/config/get_mumps_version.cpp
          CMAKE_FLAGS -DINCLUDE_DIRECTORIES:STRING=${MUMPS_INCLUDE_DIRS}
          RUN_OUTPUT_VARIABLE MUMPS_VERSION_OUTPUT)
  if ((MUMPS_VERSION_RUN_RESULT EQUAL 0) AND MUMPS_VERSION_OUTPUT)
    string(STRIP "${MUMPS_VERSION_OUTPUT}" MUMPS_VERSION)
    set(MUMPS_VERSION ${MUMPS_VERSION} CACHE STRING "MUMPS version." FORCE)
    message(STATUS "Found MUMPS version ${MUMPS_VERSION}")
  else()
    message(FATAL_ERROR "Unable to determine MUMPS version.")
  endif()
endif()
