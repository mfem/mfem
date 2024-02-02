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

list(APPEND NAVIER_COMMON_SOURCES
    navier_solver.cpp)
list(APPEND NAVIER_COMMON_HEADERS
    navier_solver.hpp)

convert_filenames_to_full_paths(NAVIER_COMMON_SOURCES)
convert_filenames_to_full_paths(NAVIER_COMMON_HEADERS)

set(NAVIER_COMMON_FILES
    EXTRA_SOURCES ${NAVIER_COMMON_SOURCES}
    EXTRA_HEADERS ${NAVIER_COMMON_HEADERS})