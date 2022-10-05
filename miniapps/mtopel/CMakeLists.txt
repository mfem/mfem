# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND SEQADIFF_COMMON_SOURCES)


list(APPEND SEQADIFF_COMMON_HEADERS)

list(APPEND PARADIFF_COMMON_SOURCES
	mtop_filters.cpp
	mtop_solvers.cpp
	mtop_coefficients.cpp
	MMA.cpp)



list(APPEND PARADIFF_COMMON_HEADERS
	mtop_filters.hpp
	mtop_solvers.hpp
	mtop_coefficients.hpp
	MMA.hpp)

convert_filenames_to_full_paths(SEQADIFF_COMMON_SOURCES)
convert_filenames_to_full_paths(SEQADIFF_COMMON_HEADERS)

set(SEQADIFF_COMMON_FILES
	EXTRA_SOURCES ${SEQADIFF_COMMON_SOURCES}
	EXTRA_HEADERS ${SEQADIFF_COMMON_HEADERS})


if(MFEM_USE_MPI)

list(APPEND PARADIFF_COMMON_SOURCES
	)
list(APPEND PARADIFF_COMMON_HEADERS
	)

convert_filenames_to_full_paths(PARADIFF_COMMON_SOURCES)
convert_filenames_to_full_paths(PARADIFF_COMMON_HEADERS)

set(PARADIFF_COMMON_FILES
	EXTRA_SOURCES ${PARADIFF_COMMON_SOURCES} ${SEQADIFF_COMMON_SOURCES}
	EXTRA_HEADERS ${PARADIFF_COMMON_HEADERS} ${SEQADIFF_COMMON_HEADERS})

message(STATUS "PARADIFF_COMMON_FILES: ${PARADIFF_COMMON_FILES}")
message(STATUS "SEQADIFF_COMMON_FILES: ${SEQADIFF_COMMON_FILES}")

add_mfem_miniapp(mtop_filt_test
     MAIN filter_test.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_filt_test01
     MAIN filter_test01.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_elast_test
     MAIN elast_solv_test.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)     

add_mfem_miniapp(mtop_compl_test
     MAIN compl_obj_test.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_top
     MAIN mtop_top.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_mbb_filt
     MAIN mtop_mbb_filt.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_mbb_nfil
     MAIN mtop_mbb_nfil.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)



add_mfem_miniapp(mtop_mul
     MAIN mtop_mul.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_lam
     MAIN mtop_lam.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_dom
     MAIN mtop_dom.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_bra
     MAIN mtop_bra.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(shoot_test
     MAIN shooting_test.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(mtop_alco
     MAIN mtop_alcoa.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

endif ()
