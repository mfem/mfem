# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# The FSI utilities (interface transfer, coefficient containers, etc.) require
# MPI (ParSubMesh). GSLIB is optional: without it, only the native transfer
# backend is available.

set(SRCS
  fsi_utils.cpp
  interface_transfer.cpp)
set(HDRS
  FilesystemHelper.hpp
  anisotropy_utils.hpp
  fsi_utils.hpp
  interface_transfer.hpp)

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

if (MFEM_USE_CUDA)
  set_property(SOURCE ${SRCS} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
  set_property(SOURCE ${SRCS} PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

mfem_add_library(mfem-common-fsi ${SRCS} ${HDRS})
target_include_directories(mfem-common-fsi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mfem-common-fsi mfem)

# Mesh utilities
add_mfem_miniapp(mesh-utils
  MAIN mesh-utils.cpp
  LIBRARIES mfem-common mfem-common-fsi)

# Tests for the interface transfer class
add_mfem_miniapp(transfer2d
  MAIN tests/transfer2d.cpp
  LIBRARIES mfem-common mfem-common-fsi)

add_mfem_miniapp(transfer3d
  MAIN tests/transfer3d.cpp
  LIBRARIES mfem-common mfem-common-fsi)
