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

list(APPEND SRCS
  mesh_extras.cpp
  fem_extras.cpp
  )
list(APPEND HDRS
  mesh_extras.hpp
  fem_extras.hpp
  mfem-common.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pfem_extras.cpp
    dist_solver.cpp)
  list(APPEND HDRS
    pfem_extras.hpp
    dist_solver.hpp)
endif()

if (MFEM_USE_CUDA)
  set_property(SOURCE ${SRCS} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
  set_property(SOURCE ${SRCS} PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(MINIAPPS_COMMON_SOURCES ${MINIAPPS_COMMON_SOURCES} ${SRCS} PARENT_SCOPE)
set(MINIAPPS_COMMON_HEADERS ${MINIAPPS_COMMON_HEADERS} ${HDRS} PARENT_SCOPE)

mfem_add_library(mfem-common ${SRCS} ${HDRS})
target_link_libraries(mfem-common mfem)
set_target_properties(mfem-common PROPERTIES VERSION "${mfem_VERSION}")
set_target_properties(mfem-common PROPERTIES SOVERSION "${mfem_VERSION}")
install(TARGETS mfem-common
  ARCHIVE DESTINATION ${INSTALL_LIB_DIR}
  LIBRARY DESTINATION ${INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${INSTALL_BIN_DIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/miniapps/common
  DESTINATION ${INSTALL_INCLUDE_DIR}/mfem/miniapps
  FILES_MATCHING PATTERN "*.hpp")
