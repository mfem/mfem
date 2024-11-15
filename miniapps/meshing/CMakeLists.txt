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

set(MESH_FILES
  amr-quad-q2.mesh
  blade.mesh
  cube.mesh
  cube-tet.mesh
  icf.mesh
  jagged.mesh
  square01.mesh
  square01-tri.mesh
  stretched2D.mesh
)

# Add a target to copy the mesh files from the source directory; used by sample
# runs.
set(SRC_MESH_FILES)
foreach(MESH_FILE ${MESH_FILES})
  list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${MESH_FILE})
endforeach()
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} .
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying meshing miniapps data files ...")
add_custom_target(copy_miniapps_meshing_data DEPENDS data_is_copied)

add_mfem_miniapp(klein-bottle
  MAIN klein-bottle.cpp
  LIBRARIES mfem)

add_mfem_miniapp(mesh-explorer
  MAIN mesh-explorer.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)

add_mfem_miniapp(mobius-strip
  MAIN mobius-strip.cpp
  LIBRARIES mfem)

add_mfem_miniapp(shaper
  MAIN shaper.cpp
  LIBRARIES mfem)

add_mfem_miniapp(extruder
  MAIN extruder.cpp
  LIBRARIES mfem)

add_mfem_miniapp(mesh-optimizer
  MAIN mesh-optimizer.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)
add_dependencies(mesh-optimizer copy_miniapps_meshing_data)

add_mfem_miniapp(mesh-quality
  MAIN mesh-quality.cpp
  ${MFEM_MINIAPPS_COMMON_HEADERS}
  LIBRARIES mfem-common)

add_mfem_miniapp(minimal-surface
  MAIN minimal-surface.cpp
  LIBRARIES mfem)

add_mfem_miniapp(reflector
  MAIN reflector.cpp
  LIBRARIES mfem)

add_mfem_miniapp(toroid
  MAIN toroid.cpp
  LIBRARIES mfem)

add_mfem_miniapp(trimmer
  MAIN trimmer.cpp
  LIBRARIES mfem)

add_mfem_miniapp(twist
  MAIN twist.cpp
  LIBRARIES mfem)

add_mfem_miniapp(polar-nc
  MAIN polar-nc.cpp
  LIBRARIES mfem)

# Add serial tests.
if (MFEM_ENABLE_TESTING)
  set(SERIAL_TESTS
    mobius-strip
    klein-bottle
    toroid
    trimmer
    twist
    extruder
    mesh-optimizer
    minimal-surface
    polar-nc
    reflector
    mesh-quality
  )
  foreach(test ${SERIAL_TESTS})
    add_test(NAME ${test} COMMAND ${test} -no-vis)
  endforeach()
endif()

# Parallel apps.
if (MFEM_USE_MPI)
  add_mfem_miniapp(pmesh-optimizer
    MAIN pmesh-optimizer.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)
  add_dependencies(pmesh-optimizer copy_miniapps_meshing_data)

  add_mfem_miniapp(pmesh-fitting
    MAIN pmesh-fitting.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)
  add_dependencies(pmesh-fitting copy_miniapps_meshing_data)

  add_mfem_miniapp(fit-node-position
    MAIN fit-node-position.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)
  add_dependencies(fit-node-position copy_miniapps_meshing_data)

  add_mfem_miniapp(pminimal-surface
    MAIN pminimal-surface.cpp
    LIBRARIES mfem)

  # Add parallel tests.
  if (MFEM_ENABLE_TESTING)
    set(PARALLEL_TESTS
      pmesh-optimizer
      pmesh-fitting
      fit-node-position
      pminimal-surface
    )
    # Meshing miniapps that return MFEM_SKIP_RETURN_VALUE in some cases:
    set(SKIP_TESTS)
    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
      list(APPEND SKIP_TESTS pmesh-fitting)
    endif()

    foreach(test ${PARALLEL_TESTS})
      if (test IN_LIST SKIP_TESTS)
        continue()
      endif()
      add_test(NAME ${test}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test}> -no-vis
        ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
endif()
