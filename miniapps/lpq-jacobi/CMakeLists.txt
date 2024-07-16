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

if(MFEM_USE_MPI)
  set(MESH_FILES_FROM_MESHING
    icf.mesh
    cube.mesh
    # jagged.mesh
    # square01.mesh
    # square01-tri.mesh
    # stretched2D.mesh
    # amr-quad-q2.mesh
    # blade.mesh
    # cube-tet.mesh
  )
  
  set(MESH_FILES_FROM_GSLIB
    triple-pt-1.mesh
    triple-pt-2.mesh
  )
  
  set(MESH_FILES_FROM_DATA
    beam-tet.mesh
    square-disc-p2.mesh
    fichera-mixed-p2.mesh
  )
  
  # Add a target to copy the mesh files from the source directory; used by sample
  # runs.
  set(SRC_MESH_FILES)
  foreach(MESH_FILE ${MESH_FILES_FROM_MESHING})
    list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../meshing/${MESH_FILE})
  endforeach()
  
  foreach(MESH_FILE ${MESH_FILES_FROM_GSLIB})
    list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../gslib/${MESH_FILE})
  endforeach()
  
  foreach(MESH_FILE ${MESH_FILES_FROM_DATA})
    list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../data/${MESH_FILE})
  endforeach()
  
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/meshes)
  
  add_custom_command(OUTPUT data_is_copied
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} ./meshes
    COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
    COMMENT "Copying meshing miniapps data files ...")
  add_custom_target(copy_meshes DEPENDS data_is_copied)
  
  add_mfem_miniapp(lpq-jacobi
    MAIN lpq-jacobi.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    EXTRA_HEADERS lpq-jacobi.hpp
    LIBRARIES mfem-common mfem)
  add_dependencies(lpq-jacobi copy_meshes)

  #   add_mfem_miniapp(ml-lpq-jacobi
  #     MAIN ml-lpq-jacobi.cpp
  #     ${MFEM_MINIAPPS_COMMON_HEADERS}
  #     EXTRA_HEADERS lpq-jacobi.hpp
  #     LIBRARIES mfem-common mfem)
  #   add_dependencies(ml-lpq-jacobi copy_meshes)
endif()


# TODO(Gabriel): Add parallel code and tests.
# # Parallel apps.
# if (MFEM_USE_MPI)
#   add_mfem_miniapp(plpq-jacobi
#     MAIN plpq-jacobi.cpp
#     ${MFEM_MINIAPPS_COMMON_HEADERS}
#     LIBRARIES mfem-common mfem)
#   add_dependencies(plpq-jacobi copy_meshes)
# 
#   # Add parallel tests.
#   if (MFEM_ENABLE_TESTING)
#     set(PARALLEL_TESTS
#       plpq-jacobi
#     )
#     # TODO(Gabriel): Check this
#     # Meshing miniapps that return MFEM_SKIP_RETURN_VALUE in some cases:
#     set(SKIP_TESTS)
# 
#     foreach(test ${PARALLEL_TESTS})
#       if (test IN_LIST SKIP_TESTS)
#         continue()
#       endif()
#       add_test(NAME ${test}_np=${MFEM_MPI_NP}
#         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
#         ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test}> -no-vis
#         ${MPIEXEC_POSTFLAGS})
#     endforeach()
#   endif()
# endif()
