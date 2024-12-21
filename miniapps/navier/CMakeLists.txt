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

if (MFEM_USE_MPI)
  list(APPEND NAVIER_COMMON_SOURCES
      navier_solver.cpp)
  list(APPEND NAVIER_COMMON_HEADERS
      navier_solver.hpp)

  convert_filenames_to_full_paths(NAVIER_COMMON_SOURCES)
  convert_filenames_to_full_paths(NAVIER_COMMON_HEADERS)

  set(NAVIER_COMMON_FILES
      EXTRA_SOURCES ${NAVIER_COMMON_SOURCES}
      EXTRA_HEADERS ${NAVIER_COMMON_HEADERS})

  add_mfem_miniapp(navier_mms
    MAIN navier_mms.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_kovasznay
    MAIN navier_kovasznay.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_kovasznay_vs
    MAIN navier_kovasznay_vs.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_tgv
    MAIN navier_tgv.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_shear
    MAIN navier_shear.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_3dfoc
    MAIN navier_3dfoc.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_turbchan
    MAIN navier_turbchan.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  if (MFEM_USE_GSLIB)
    add_mfem_miniapp(navier_cht
      MAIN navier_cht.cpp
      ${NAVIER_COMMON_FILES}
      LIBRARIES mfem)

    LIST(APPEND NAVIER_MESH_FILES
        fluid-cht.mesh
        solid-cht.mesh)
    foreach(MESH_FILE ${NAVIER_MESH_FILES})
        add_custom_command(
          TARGET navier_cht POST_BUILD #OUTPUT ${MESH_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${MESH_FILE} ${MESH_FILE}
          COMMENT "copy: ${MESH_FILE}")
    endforeach()
  endif ()

  if (MFEM_ENABLE_TESTING)
    foreach (test "mms" "kovasznay" "tgv")
      add_test(NAME navier_${test}_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:navier_${test}> -cr -no-vis
        ${MPIEXEC_POSTFLAGS})
    endforeach()
endif()
endif ()
