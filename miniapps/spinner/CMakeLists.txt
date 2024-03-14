# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND SPINNER_SOURCES
    spinner-lts.cpp
    cell-solver.hpp)
list(APPEND SPINNER_HEADERS
    cell-solver.hpp
    new-kelly-estimator.hpp)

convert_filenames_to_full_paths(SPINNER_SOURCES)
convert_filenames_to_full_paths(SPINNER_HEADERS)

set(SPINNER_COMMON_FILES
    EXTRA_SOURCES ${SPINNER_SOURCES}
    EXTRA_HEADERS ${SPINNER_HEADERS})

add_mfem_miniapp(spinner-lts
    MAIN spinner-lts.cpp
    ${SPINNER_COMMON_FILES}
    LIBRARIES mfem)
