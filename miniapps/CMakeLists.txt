# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
add_subdirectory(common)
add_subdirectory(electromagnetics)
add_subdirectory(meshing)
add_subdirectory(performance)
add_subdirectory(tools)
add_subdirectory(toys)
add_subdirectory(nurbs)
add_subdirectory(gslib)
