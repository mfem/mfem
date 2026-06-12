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
#   - SAMRAI_LIBRARIES
#   - SAMRAI_INCLUDE_DIRS

if (SAMRAI_FOUND OR TARGET SAMRAI)
  return()
endif()

if (MFEM_FETCH_SAMRAI OR MFEM_FETCH_TPLS)
  set(SAMRAI_FETCH_TAG v-4-5-1)
  set(SAMRAI_CMAKE_OPTIONS "")
  list(APPEND SAMRAI_CMAKE_OPTIONS -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
                                   -DENABLE_HDF5:BOOL=OFF)
  message(STATUS "Will fetch SAMRAI ${SAMRAI_FETCH_TAG}")
  set(SAMRAI_INSTALL ${CMAKE_BINARY_DIR}/fetch/samrai)
  include(ExternalProject)
  ExternalProject_Add(samrai
    GIT_REPOSITORY https://github.com/llnl/SAMRAI.git
    GIT_TAG ${SAMRAI_FETCH_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    UPDATE_DISCONNECTED TRUE
    PREFIX ${SAMRAI_INSTALL}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${SAMRAI_INSTALL} ${SAMRAI_CMAKE_OPTIONS})

  add_library(SAMRAI_tbox STATIC IMPORTED)
  target_link_libraries(SAMRAI_tbox INTERFACE MPI_CXX)

  add_library(SAMRAI_hier STATIC IMPORTED)
  target_link_libraries(SAMRAI_hier INTERFACE SAMRAI_tbox MPI_CXX)

  add_library(SAMRAI_xfer STATIC IMPORTED)
  target_link_libraries(SAMRAI_xfer INTERFACE SAMRAI_hier SAMRAI_tbox)

  foreach(SAMRAI_library SAMRAI_pdat SAMRAI_mesh)
    add_library(${SAMRAI_library} STATIC IMPORTED)
    target_link_libraries(${SAMRAI_library} INTERFACE SAMRAI_hier SAMRAI_tbox SAMRAI_xfer)
  endforeach()

  foreach(SAMRAI_library SAMRAI_algs SAMRAI_appu SAMRAI_geom SAMRAI_math)
    add_library(${SAMRAI_library} STATIC IMPORTED)
    target_link_libraries(${SAMRAI_library} INTERFACE SAMRAI_hier SAMRAI_pdat SAMRAI_tbox)
  endforeach()

  add_library(SAMRAI_solv STATIC IMPORTED)
  target_link_libraries(SAMRAI_solv INTERFACE SAMRAI_hier SAMRAI_math SAMRAI_tbox SAMRAI_xfer)

  set(SAMRAI_LIBRARIES SAMRAI_algs SAMRAI_appu SAMRAI_geom SAMRAI_hier
                       SAMRAI_pdat SAMRAI_math SAMRAI_mesh SAMRAI_solv
                       SAMRAI_tbox SAMRAI_xfer)

  foreach(SAMRAI_library ${SAMRAI_LIBRARIES})
    add_dependencies(${SAMRAI_library} samrai)
    set_target_properties(${SAMRAI_library} PROPERTIES
      IMPORTED_LOCATION ${SAMRAI_INSTALL}/lib/lib${SAMRAI_library}.a)
  endforeach()

  set(SAMRAI_INCLUDE_DIRS ${SAMRAI_INSTALL}/include)

else()
  find_package(SAMRAI REQUIRED CONFIG)
endif()
