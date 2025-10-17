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

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

set(MINIAPPS_COMMON_SOURCES "")
set(MINIAPPS_COMMON_HEADERS "")
if (MFEM_USE_SUNDIALS)
   add_subdirectory(adjoint)
endif()
add_subdirectory(common)
add_subdirectory(electromagnetics)
add_subdirectory(navier)
add_subdirectory(meshing)
add_subdirectory(performance)
add_subdirectory(tools)
add_subdirectory(toys)
add_subdirectory(nurbs)
add_subdirectory(gslib)
add_subdirectory(spde)
add_subdirectory(solvers)
add_subdirectory(shifted)
add_subdirectory(mtop)
add_subdirectory(multidomain)
add_subdirectory(autodiff)
add_subdirectory(parelag)
add_subdirectory(tribol)
add_subdirectory(hooke)
add_subdirectory(dpg)
add_subdirectory(hdiv-linear-solver)
add_subdirectory(dfem)
add_subdirectory(diag-smoothers)
