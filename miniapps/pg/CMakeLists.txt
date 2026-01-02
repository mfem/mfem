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

list(APPEND PG_SOURCES
    legendre.cpp)
list(APPEND PG_HEADERS
    legendre.hpp
    point_pg_solver.hpp
    schur_prec.hpp
)

convert_filenames_to_full_paths(PG_SOURCES)
convert_filenames_to_full_paths(PG_HEADERS)

set(PG_COMMON_FILES
    EXTRA_SOURCES ${PG_SOURCES}
    EXTRA_HEADERS ${PG_HEADERS})

# Add targets to copy rc_* files from the source directory
set(PETSC_RC_FILES "")
list(APPEND PETSC_RC_FILES
    rc_bddc
)

foreach(RC_FILE ${PETSC_RC_FILES})
  add_custom_command(OUTPUT ${RC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/${RC_FILE} ${RC_FILE}
    COMMENT "copy ${RC_FILE}")
endforeach()
add_custom_target(copy_pg_petsc_rc_files DEPENDS ${PETSC_RC_FILES}
  COMMENT "Copying PETSC example rc files ...")

add_mfem_miniapp(obstacle
  MAIN obstacle.cpp
  ${PG_COMMON_FILES}
  LIBRARIES mfem)

add_mfem_miniapp(par_obstacle
  MAIN par_obstacle.cpp
  ${PG_COMMON_FILES}
  LIBRARIES mfem)

add_mfem_miniapp(par_obstacle_mixed
  MAIN par_obstacle_mixed.cpp
  ${PG_COMMON_FILES}
  LIBRARIES mfem)
add_dependencies(par_obstacle_mixed copy_pg_petsc_rc_files)

add_mfem_miniapp(pos_proj
  MAIN pos_proj.cpp
  ${PG_COMMON_FILES}
  LIBRARIES mfem)

