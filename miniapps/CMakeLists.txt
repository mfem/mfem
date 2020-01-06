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

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

set(MINIAPP_COMMON_HEADERS)
add_subdirectory(common)
add_subdirectory(electromagnetics)
add_subdirectory(meshing)
add_subdirectory(performance)
add_subdirectory(tools)
add_subdirectory(nurbs)
add_subdirectory(gslib)
