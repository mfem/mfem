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

# Find PASTIX.
# Defines the following variables:
#   - PASTIX_FOUND
#   - PASTIX_LIBRARIES
#   - PASTIX_INCLUDE_DIRS

# Pastix lib requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.
# Pastix requires SCOTCH or METIS (partitioning and reordering tools) as well

if (PASTIX_INCLUDE_DIRS AND PASTIX_LIBRARIES)
  set(PASTIX_FIND_QUIETLY TRUE)
endif (PASTIX_INCLUDE_DIRS AND PASTIX_LIBRARIES)

find_path(PASTIX_INCLUDE_DIRS
  NAMES
  pastix.h
  PATHS
  ${PASTIX_DIR}/include
  ${INCLUDE_INSTALL_DIR}
)

# Three separate libraries - pastix, spm, pastix_kernels
find_library(PASTIX_LIBRARIES pastix PATHS ${PASTIX_DIR}/lib ${LIB_INSTALL_DIR})
find_library(PASTIX_SPM_LIBRARIES spm PATHS ${PASTIX_DIR}/lib ${LIB_INSTALL_DIR})
find_library(PASTIX_KERNEL_LIBRARIES pastix_kernels PATHS ${PASTIX_DIR}/lib ${LIB_INSTALL_DIR})
list(APPEND PASTIX_LIBRARIES ${PASTIX_SPM_LIBRARIES})
list(APPEND PASTIX_LIBRARIES ${PASTIX_KERNEL_LIBRARIES})

if(EXISTS ${PASTIX_DIR}/examples/Makefile)
  file(STRINGS ${PASTIX_DIR}/examples/Makefile
    PASTIX_VARIABLES NEWLINE_CONSUME)
else()
  message(SEND_ERROR "PaStiX examples not found - needed to determine its TPLs")
endif()

string(REGEX MATCH "EXTRALIBS= [^\n\r]*" PASTIX_EXTRALIBS ${PASTIX_VARIABLES})
string(REPLACE "EXTRALIBS= " "" PASTIX_EXTRALIBS ${PASTIX_EXTRALIBS})
string(STRIP ${PASTIX_EXTRALIBS} PASTIX_EXTRALIBS)

list(APPEND PASTIX_LIBRARIES ${PASTIX_EXTRALIBS})

find_package(PkgConfig REQUIRED)
set(OLD_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH "${PASTIX_DIR}/lib/pkgconfig")
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
pkg_check_modules(PASTIX_PC REQUIRED pastix)
list(APPEND PASTIX_LIBRARIES ${PASTIX_PC_LDFLAGS})
set(CMAKE_PREFIX_PATH ${OLD_PREFIX_PATH})


if(UNIX AND NOT APPLE)
  find_package(Threads)
  list(APPEND PASTIX_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PASTIX DEFAULT_MSG
                                  PASTIX_INCLUDE_DIRS PASTIX_LIBRARIES)

mark_as_advanced(PASTIX_INCLUDES PASTIX_LIBRARIES)
