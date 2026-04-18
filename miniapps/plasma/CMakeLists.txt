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

if (MFEM_USE_MPI)
  list(APPEND PLASMA_SRCS
  vis_object.cpp)

  list(APPEND PLASMA_HDRS
  plasma.hpp
  vis_object.hpp)

  convert_filenames_to_full_paths(PLASMA_SRCS)
  convert_filenames_to_full_paths(PLASMA_HDRS)

  set(MINIAPPS_PLASMA_SOURCES ${MINIAPPS_PLASMA_SOURCES} ${PLASMA_SRCS} PARENT_SCOPE)
  set(MINIAPPS_PLASMA_HEADERS ${MINIAPPS_PLASMA_HEADERS} ${PLASMA_HDRS} PARENT_SCOPE)

  mfem_add_library(mfem-plasma ${PLASMA_SRCS} ${PLASMA_HDRS})
  target_link_libraries(mfem-plasma mfem-common mfem)
  set_target_properties(mfem-plasma PROPERTIES VERSION "${mfem_VERSION}")
  set_target_properties(mfem-plasma PROPERTIES SOVERSION "${mfem_VERSION}")
  install(TARGETS mfem-plasma
    ARCHIVE DESTINATION ${INSTALL_LIB_DIR}
    LIBRARY DESTINATION ${INSTALL_LIB_DIR}
    RUNTIME DESTINATION ${INSTALL_BIN_DIR})
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/miniapps/plasma
    DESTINATION ${INSTALL_INCLUDE_DIR}/mfem/miniapps
    FILES_MATCHING PATTERN "*.hpp")
endif()

add_subdirectory(electromagnetics)
add_subdirectory(pic)
