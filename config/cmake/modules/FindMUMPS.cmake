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
elseif(MFEM_USE_SINGLE)
  set(_mumps_header smumps_c.h)
  set(_mumps_lib smumps)
endif()

mfem_find_package(MUMPS MUMPS MUMPS_DIR
  "include" ${_mumps_header} "lib" ${_mumps_lib}
  "Paths to headers required by MUMPS."
  "Libraries required by MUMPS."
  ADD_COMPONENT mumps_common "include" ${_mumps_header} "lib" mumps_common
  ADD_COMPONENT pord "include" ${_mumps_header} "lib" pord)

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
