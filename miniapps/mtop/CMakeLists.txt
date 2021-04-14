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

list(APPEND SEQMTOP_COMMON_SOURCES
        mtop_integrators.cpp)
list(APPEND SQEMTOP_COMMON_HEADERS
        mtop_integrators.hpp)

convert_filenames_to_full_paths(SEQMTOP_COMMON_SOURCES)
convert_filenames_to_full_paths(SEQMTOP_COMMON_HEADERS)

set(SEQMTOP_COMMON_FILES
	EXTRA_SOURCES ${SEQMTOP_COMMON_SOURCES}
	EXTRA_HEADERS ${SEQMTOP_COMMON_HEADERS})

add_mfem_miniapp(seqheat
	MAIN seqheat.cpp
	${SEQMTOP_COMMON_FILES}
	LIBRARIES mfem)

if (MFEM_USE_MPI)
list(APPEND MTOP_COMMON_SOURCES
        )
list(APPEND MTOP_COMMON_HEADERS
        )

convert_filenames_to_full_paths(MTOP_COMMON_SOURCES)
convert_filenames_to_full_paths(MTOP_COMMON_HEADERS)

set(MTOP_COMMON_FILES
          EXTRA_SOURCES ${MTOP_COMMON_SOURCES} ${SEQMTOP_COMMON_SOURCES}
          EXTRA_HEADERS ${MTOP_COMMON_HEADERS} ${SEQMTOP_COMMON_HEADERS})

add_mfem_miniapp(parheat
     MAIN parheat.cpp
     ${MTOP_COMMON_FILES} 
     LIBRARIES mfem)

add_mfem_miniapp(paradvec
     MAIN paradvec.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(meshr01
     MAIN meshr01.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(meshr02
     MAIN meshr02.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(meshr03
     MAIN meshr03.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(meshr04
     MAIN meshr04.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(meshr05
     MAIN meshr05.cpp
     ${MTOP_COMMON_FILES}
     LIBRARIES mfem)

endif ()
