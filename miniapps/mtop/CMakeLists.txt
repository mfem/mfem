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
	paramnonlinearform.cpp
        mtop_integrators.cpp)

list(APPEND SEQMTOP_COMMON_HEADERS
	paramnonlinearform.hpp
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

if(MFEM_USE_MPI)

list(APPEND PARMTOP_COMMON_SOURCES
	pparamnonlinearform.cpp)
list(APPEND PARMTOP_COMMON_HEADERS
	pparamnonlinearform.hpp)

convert_filenames_to_full_paths(PARMTOP_COMMON_SOURCES)
convert_filenames_to_full_paths(PARMTOP_COMMON_HEADERS)

set(PARMTOP_COMMON_FILES
	EXTRA_SOURCES ${PARMTOP_COMMON_SOURCES} ${SEQMTOP_COMMON_SOURCES}
	EXTRA_HEADERS ${PARMTOP_COMMON_HEADERS} ${SEQMTOP_COMMON_HEADERS})

message(STATUS "PARMTOP_COMMON_FILES: ${PARMTOP_COMMON_FILES}")
message(STATUS "SEQMTOP_COMMON_FILES: ${SEQMTOP_COMMON_FILES}")

add_mfem_miniapp(parheat
     MAIN parheat.cpp
     ${PARMTOP_COMMON_FILES}
     LIBRARIES mfem)

endif ()
