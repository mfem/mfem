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

if (MFEM_USE_MPI)
  list(APPEND DIST_COMMON_SOURCES
  	distfunction.cpp nldist.cpp)
  list(APPEND DIST_COMMON_HEADERS
	distfunction.hpp nldist.hpp)

convert_filenames_to_full_paths(DIST_COMMON_SOURCES)
convert_filenames_to_full_paths(DIST_COMMON_HEADERS)

set(DIST_COMMON_FILES
	EXTRA_SOURCES ${DIST_COMMON_SOURCES}
	EXTRA_HEADERS ${DIST_COMMON_HEADERS})

  add_mfem_miniapp(ex1p1
    MAIN ex1p.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(distance1
    MAIN distance.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(dist_solv
    MAIN dist_solver.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem)


endif ()
