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

list(APPEND SEQCUT_COMMON_SOURCES
	cut_marking.cpp
	my_integrators.cpp)

list(APPEND SEQCUT_COMMON_HEADERS
	cut_marking.hpp
	my_integrators.hpp)

convert_filenames_to_full_paths(SEQCUT_COMMON_SOURCES)
convert_filenames_to_full_paths(SEQCUT_COMMON_HEADERS)

set(SEQCUT_COMMON_FILES
	EXTRA_SOURCES ${SEQCUT_COMMON_SOURCES}
	EXTRA_HEADERS ${SEQCUT_COMMON_HEADERS})

add_mfem_miniapp(seqdiff
	MAIN diffusion.cpp
	${SEQCUT_COMMON_FILES}
	LIBRARIES mfem)

add_mfem_miniapp(seqdiffn
	MAIN diffusion_nitsche.cpp
	${SEQCUT_COMMON_FILES}
        LIBRARIES mfem)

add_mfem_miniapp(seqstokes
	MAIN stokes.cpp
	${SEQCUT_COMMON_FILES}
        LIBRARIES mfem)

add_mfem_miniapp(seq_mark_test
        MAIN seq_mark_test.cpp
        ${SEQCUT_COMMON_FILES}
        LIBRARIES mfem)

add_mfem_miniapp(seq_diffusion_cut
        MAIN diffusion_cut.cpp
        ${SEQCUT_COMMON_FILES}
        LIBRARIES mfem)

add_mfem_miniapp(seq_stokes_cut
	MAIN stokes_cut.cpp
	${SEQCUT_COMMON_FILES}
	LIBRARIES mfem)

add_mfem_miniapp(seq_vector
	MAIN vector_diffusion.cpp
	${SEQCUT_COMMON_FILES}
	LIBRARIES mfem)

add_mfem_miniapp(seq_vector_cut
	MAIN vector_diffusion_cut.cpp
	${SEQCUT_COMMON_FILES}
	LIBRARIES mfem)

if(MFEM_USE_MPI)

list(APPEND PARCUT_COMMON_SOURCES)
list(APPEND PARCUT_COMMON_HEADERS)

convert_filenames_to_full_paths(PARCUT_COMMON_SOURCES)
convert_filenames_to_full_paths(PARCUT_COMMON_HEADERS)

set(PARCUT_COMMON_FILES
	EXTRA_SOURCES ${PARCUT_COMMON_SOURCES} ${SEQCUT_COMMON_SOURCES}
	EXTRA_HEADERS ${PARCUT_COMMON_HEADERS} ${SEQCUT_COMMON_HEADERS})

# message(STATUS "PARCUT_COMMON_FILES: ${PARCUT_COMMON_FILES}")
# message(STATUS "SEQCUT_COMMON_FILES: ${SEQCUT_COMMON_FILES}")

add_mfem_miniapp(par_diffusion_cut
        MAIN par_diffusion_cut.cpp
        ${PARCUT_COMMON_FILES}
        LIBRARIES mfem)

add_mfem_miniapp(par_diffusion_cut_3D
	MAIN par_diffusion_cut_3D.cpp
        ${PARCUT_COMMON_FILES}
        LIBRARIES mfem)


endif ()
