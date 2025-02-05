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
    amr-quad.mesh
    ref-cube.mesh
    ref-square.mesh
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
  
  add_custom_command(OUTPUT data_is_copied
    COMMAND ${CMAKE_COMMAND} -E make_directory meshes
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} ./meshes
    COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
    COMMENT "Copying meshing miniapps data files ...")
  add_custom_target(copy_meshes DEPENDS data_is_copied)

  add_custom_target(clean_meshes
    COMMAND ${CMAKE_COMMAND} -E remove_directory meshes
    COMMENT "Cleaning up meshes directory..."
  )

  add_custom_target(clean_common
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning common directory..."
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/../common clean
    COMMENT "Cleaning common directory..."
  )

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES meshes)
  
  # Define miniapps
  set(MINIAPPS
    lpq-jacobi
    mg-lpq-jacobi
    abs-l1-jacobi
    mg-abs-l1-jacobi
  )

  # Add miniapps
  foreach(APP ${MINIAPPS})
    add_mfem_miniapp(${APP}
      MAIN ${APP}.cpp
      ${MFEM_MINIAPPS_COMMON_HEADERS}
      EXTRA_SOURCES lpq-common.cpp
      EXTRA_HEADERS lpq-common.hpp
      LIBRARIES mfem-common mfem
    )
    add_dependencies(${APP} copy_meshes)
  endforeach()

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME lpq-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lpq-jacobi> -m meshes/cube.mesh -rs 1 -rp 1
      -s 1 -i 1 -p 1.5 -q 0.75 -pc 1 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME abs-l1-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:abs-l1-jacobi> -m meshes/cube.mesh -rs 1 -rp 1
      -s 1 -i 1 -a 4 -pc 1 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME mg-lpq-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mg-lpq-jacobi> -m meshes/cube.mesh -rs 1 -rp 1
      -ol 1 -gl 1 -s 1 -i 1 -p 1.5 -q 0.75 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME mg-abs-l1-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mg-abs-l1-jacobi> -m meshes/cube.mesh -rs 1 -rp 1
      -ol 1 -gl 1 -s 1 -i 1 -a 4 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})
  endif()

  # Add custom target for cleaning everything
  add_custom_target(clean_all
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory meshes
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_SOURCE_DIR}/../common clean
    COMMENT "Cleaning all build artifacts and meshes..."
  )
endif()
