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

list(APPEND SEQADIFF_COMMON_SOURCES
  ad_native.cpp
  logger.cpp
  pg.cpp
)

list(APPEND SEQADIFF_COMMON_HEADERS
	tadvector.hpp
	taddensemat.hpp
	admfem.hpp
  ad_intg.hpp
  ad_native.hpp
  logger.hpp
  pg.hpp
  tools.hpp
)

convert_filenames_to_full_paths(SEQADIFF_COMMON_SOURCES)
convert_filenames_to_full_paths(SEQADIFF_COMMON_HEADERS)

set(SEQADIFF_COMMON_FILES
	EXTRA_SOURCES ${SEQADIFF_COMMON_SOURCES}
	EXTRA_HEADERS ${SEQADIFF_COMMON_HEADERS})

add_mfem_miniapp(ad_ex0 MAIN ad_ex0.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
add_mfem_miniapp(ad_ex1 MAIN ad_ex1.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
add_mfem_miniapp(ad_ex2 MAIN ad_ex2.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
add_mfem_miniapp(ad_ex3 MAIN ad_ex3.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
if (MFEM_USE_MUMPS)
  add_mfem_miniapp(ad_ex4 MAIN ad_ex4.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
  add_mfem_miniapp(ad_ex5 MAIN ad_ex5.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
endif ()
if (MFEM_USE_MPI)
  add_mfem_miniapp(ad_ex6 MAIN ad_ex6.cpp ${SEQADIFF_COMMON_FILES} LIBRARIES mfem)
endif ()

