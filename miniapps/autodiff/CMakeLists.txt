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

list(APPEND SEQADIFF_COMMON_SOURCES
        stokes.cpp
	advdiff.cpp)

list(APPEND SEQADIFF_COMMON_HEADERS
        fdual.hpp
	tadvector.hpp
	taddensemat.hpp
	adnonlininteg.hpp
	admfem.hpp
	stokes.hpp
	advdiff.hpp)



convert_filenames_to_full_paths(SEQADIFF_COMMON_SOURCES)
convert_filenames_to_full_paths(SEQADIFF_COMMON_HEADERS)

set(SEQADIFF_COMMON_FILES
	EXTRA_SOURCES ${SEQADIFF_COMMON_SOURCES}
	EXTRA_HEADERS ${SEQADIFF_COMMON_HEADERS})

add_mfem_miniapp(seqadiff
	MAIN seq_example.cpp
	${SEQADIFF_COMMON_FILES}
	LIBRARIES mfem)

add_mfem_miniapp(seqtest
	MAIN seq_test.cpp
	${SEQADIFF_COMMON_FILES}
        LIBRARIES mfem)

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

add_mfem_miniapp(paradiff
     MAIN par_example.cpp
     ${PARADIFF_COMMON_FILES}
     LIBRARIES mfem)

add_mfem_miniapp(stokes
     MAIN stokes_test.cpp
     LIBRARIES mfem)

add_mfem_miniapp(advdiff
     MAIN advdiff_test.cpp
     LIBRARIES mfem)


endif ()
