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

if(MFEM_USE_MPI)

list(APPEND PARMTOP_COMMON_SOURCES
	mtop_solvers.cpp)
list(APPEND PARMTOP_COMMON_HEADERS
	mtop_solvers.hpp)

convert_filenames_to_full_paths(PARMTOP_COMMON_SOURCES)
convert_filenames_to_full_paths(PARMTOP_COMMON_HEADERS)

set(PARMTOP_COMMON_FILES
	EXTRA_SOURCES ${PARMTOP_COMMON_SOURCES} ${SEQMTOP_COMMON_SOURCES}
	EXTRA_HEADERS ${PARMTOP_COMMON_HEADERS} ${SEQMTOP_COMMON_HEADERS})

# message(STATUS "PARMTOP_COMMON_FILES: ${PARMTOP_COMMON_FILES}")
# message(STATUS "SEQMTOP_COMMON_FILES: ${SEQMTOP_COMMON_FILES}")

add_mfem_miniapp(test_isoel
     MAIN test_iso_elast.cpp
     ${PARMTOP_COMMON_FILES}
     LIBRARIES mfem)

endif ()
