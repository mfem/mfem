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

# Defines the following variables if fetching of TPLs is disabled (default):
#   - GSLIB_FOUND
#   - GSLIB_LIBRARIES
#   - GSLIB_INCLUDE_DIRS
# otherwise, the following are defined:
#   - GSLIB (imported library target)

if (MFEM_FETCH_GSLIB OR MFEM_FETCH_TPLS)
  enable_language(C)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
  set(GSLIB_FETCH_VERSION 1.0.9)
  set(GSLIB_C_FLAGS ${CMAKE_C_FLAGS_${BUILD_TYPE}})
  if (CMAKE_C_FLAGS)
    set(GSLIB_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE}}")
  endif()
  if (BUILD_SHARED_LIBS)
    set(GSLIB_C_FLAGS "${GSLIB_C_FLAGS} -fPIC")
  endif()
  add_library(GSLIB STATIC IMPORTED)
  # define external project and create future include directory so it is present
  # to pass CMake checks at end of MFEM configuration step
  message(STATUS "Will fetch GSLIB ${GSLIB_FETCH_VERSION} to be built with ${GSLIB_C_FLAGS}")
  set(PREFIX ${CMAKE_BINARY_DIR}/fetch/gslib)
  include(ExternalProject)
  ExternalProject_Add(gslib
    GIT_REPOSITORY https://github.com/Nek5000/gslib
    GIT_TAG v${GSLIB_FETCH_VERSION}
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
    PREFIX ${PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd ${PREFIX}/src/gslib && $(MAKE) clean && $(MAKE) DESTDIR=${PREFIX} MPI=$<BOOL:${MFEM_USE_MPI}> "CFLAGS= ${GSLIB_C_FLAGS}"
    INSTALL_COMMAND "")
  file(MAKE_DIRECTORY ${PREFIX}/include)
  # set imported library target properties
  add_dependencies(GSLIB gslib)
  set_target_properties(GSLIB PROPERTIES
    IMPORTED_LOCATION ${PREFIX}/lib/libgs.a
    INTERFACE_INCLUDE_DIRECTORIES ${PREFIX}/include)
  return()
endif()

include(MfemCmakeUtilities)
mfem_find_package(GSLIB GSLIB GSLIB_DIR "include" gslib.h "lib" gs
  "Paths to headers required by GSLIB." "Libraries required by GSLIB.")
