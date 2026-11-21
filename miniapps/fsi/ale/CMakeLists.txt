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

# ALE mesh deformation library
set(ALE_DEFORMATION_SOURCES
  mesh_deformation.cpp
  mesh_laplacian.cpp
  mesh_linear_elasticity.cpp
  local_stiffening_coefficient.cpp)
set(ALE_DEFORMATION_HEADERS
  mesh_deformation.hpp
  mesh_laplacian.hpp
  mesh_linear_elasticity.hpp
  local_stiffening_coefficient.hpp)

convert_filenames_to_full_paths(ALE_DEFORMATION_SOURCES)
convert_filenames_to_full_paths(ALE_DEFORMATION_HEADERS)

if (MFEM_USE_CUDA)
  set_property(SOURCE ${ALE_DEFORMATION_SOURCES} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
  set_property(SOURCE ${ALE_DEFORMATION_SOURCES}
    PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

mfem_add_library(ale-deformation
  ${ALE_DEFORMATION_SOURCES} ${ALE_DEFORMATION_HEADERS})
target_include_directories(ale-deformation PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/miniapps/common)
target_link_libraries(ale-deformation mfem-common-fsi mfem-common mfem)
