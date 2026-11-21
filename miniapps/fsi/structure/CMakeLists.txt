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

# Elasticity (structure) solver library
set(ELASTICITY_SOURCES
  elasticity_operator.cpp
  elasticity_solver.cpp)
set(ELASTICITY_HEADERS
  elasticity_operator.hpp
  elasticity_solver.hpp
  elasticity_utils.hpp)

convert_filenames_to_full_paths(ELASTICITY_SOURCES)
convert_filenames_to_full_paths(ELASTICITY_HEADERS)

if (MFEM_USE_CUDA)
  set_property(SOURCE ${ELASTICITY_SOURCES} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
  set_property(SOURCE ${ELASTICITY_SOURCES}
    PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

mfem_add_library(elasticity ${ELASTICITY_SOURCES} ${ELASTICITY_HEADERS})
target_include_directories(elasticity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(elasticity mfem-common-fsi mfem-common mfem)

add_mfem_miniapp(shell1d
  MAIN shell1d.cpp
  LIBRARIES mfem-common mfem-common-fsi)

add_mfem_miniapp(shell1d-fluid
  MAIN shell1d-fluid.cpp
  LIBRARIES mfem-common mfem-common-fsi)
