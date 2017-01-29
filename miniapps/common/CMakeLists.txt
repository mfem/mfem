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

set(MFEM_MINIAPPS_COMMON_SOURCES
  mesh_extras.cpp
  fem_extras.cpp
  )
set(MFEM_MINIAPPS_COMMON_HEADERS
  mesh_extras.hpp
  fem_extras.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND MFEM_MINIAPPS_COMMON_SOURCES
    pfem_extras.cpp)
  list(APPEND MFEM_MINIAPPS_COMMON_HEADERS
    pfem_extras.hpp)
endif()

add_library(mfem_miniapps_common ${MFEM_MINIAPPS_COMMON_SOURCES}
            ${MFEM_MINIAPPS_COMMON_HEADERS})
