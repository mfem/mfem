# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

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
    pfem_extras.cpp)
  list(APPEND HDRS
    pfem_extras.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(MINIAPPS_COMMON_SOURCES ${MINIAPPS_COMMON_SOURCES} ${SRCS} PARENT_SCOPE)
set(MINIAPPS_COMMON_HEADERS ${MINIAPPS_COMMON_HEADERS} ${HDRS} PARENT_SCOPE)

add_library(mfem-common ${SRCS} ${HDRS})
target_link_libraries(mfem-common mfem)
