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

# Decide headers/libs by MFEM precision
if (MFEM_USE_DOUBLE)
  set(_rmumps_header dmumps_c.h)
  set(_rmumps_lib    dmumps)
  set(_cmumps_header zmumps_c.h)
  set(_cmumps_lib    zmumps)
elseif (MFEM_USE_SINGLE)
  set(_rmumps_header smumps_c.h)
  set(_rmumps_lib    smumps)
  set(_cmumps_header cmumps_c.h)
  set(_cmumps_lib    cmumps)
endif()

# Choose which header/lib mfem_find_package should use as the "primary" one.
# If both enabled, prefer the real one as primary (either is fine).
if (MFEM_USE_MUMPS)
  set(_mumps_header ${_rmumps_header})
  set(_mumps_lib    ${_rmumps_lib})
elseif (MFEM_USE_COMPLEX_MUMPS)
  set(_mumps_header ${_cmumps_header})
  set(_mumps_lib    ${_cmumps_lib})
else()
  # Should not happen in practice because FindMUMPS is only called when enabled,
  set(_mumps_header ${_rmumps_header})
  set(_mumps_lib    ${_rmumps_lib})
endif()

mfem_find_package(MUMPS MUMPS MUMPS_DIR
  "include" ${_mumps_header} "lib" ${_mumps_lib}
  "Paths to headers required by MUMPS."
  "Libraries required by MUMPS."
  ADD_COMPONENT mumps_common "include" ${_mumps_header} "lib" mumps_common
  ADD_COMPONENT pord        "include" ${_mumps_header} "lib" pord)

# If BOTH real and complex are enabled, ensure BOTH solver libs are linked.
if (MUMPS_FOUND AND MFEM_USE_MUMPS AND MFEM_USE_COMPLEX_MUMPS)
  # Find the "other" solver library and append it.
  find_library(_mfem_other_mumps_solver
    NAMES ${_cmumps_lib}   
    HINTS ${MUMPS_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)

  if (NOT _mfem_other_mumps_solver)
    # Fall back to system search
    find_library(_mfem_other_mumps_solver NAMES ${_cmumps_lib})
  endif()

  if (NOT _mfem_other_mumps_solver)
    message(FATAL_ERROR
      "MFEM_USE_MUMPS=ON and MFEM_USE_COMPLEX_MUMPS=ON, but could not find "
      "the complex solver library '${_cmumps_lib}' in MUMPS_DIR='${MUMPS_DIR}'.")
  endif()

  # Put solver libs first (important for static link order)
  # MUMPS_LIBRARIES contains the primary solver already + common + pord.
  # We prepend the other solver.
  list(INSERT MUMPS_LIBRARIES 0 ${_mfem_other_mumps_solver})
endif()

# Version detection 
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
