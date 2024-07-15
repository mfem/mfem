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

set(MESH_GF_FILES
    triple-pt-1.mesh
    triple-pt-2.mesh
    triple-pt-1.gf
    triple-pt-2.gf
)

set(SRC_MESH_GF_FILES)
foreach(MESH_GF_FILE ${MESH_GF_FILES})
  list(APPEND SRC_MESH_GF_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${MESH_GF_FILE})
endforeach()
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_GF_FILES} .
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying gslib miniapps data files ...")
add_custom_target(copy_miniapps_gslib_data DEPENDS data_is_copied)

if (MFEM_USE_GSLIB)
  add_mfem_miniapp(schwarz_ex1
    MAIN schwarz_ex1.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(field-diff
    MAIN field-diff.cpp
    LIBRARIES mfem)
  add_dependencies(field-diff copy_miniapps_gslib_data)

  add_mfem_miniapp(findpts
    MAIN findpts.cpp
    LIBRARIES mfem-common)

  add_mfem_miniapp(field-interp
    MAIN field-interp.cpp
    LIBRARIES mfem)
  add_dependencies(field-interp
    copy_miniapps_gslib_data copy_miniapps_meshing_data)

  if (MFEM_ENABLE_TESTING)
    foreach (test "schwarz_ex1" "field-diff" "findpts" "field-interp")
      add_test(NAME ${test}
        COMMAND $<TARGET_FILE:${test}> -no-vis)
    endforeach()
  endif()

  # Parallel apps.
  if (MFEM_USE_MPI)
    add_mfem_miniapp(schwarz_ex1p
      MAIN schwarz_ex1p.cpp
      LIBRARIES mfem)

    add_mfem_miniapp(pfindpts
      MAIN pfindpts.cpp
      LIBRARIES mfem)

    if (MFEM_ENABLE_TESTING)
      foreach (test "schwarz_ex1p" "pfindpts")
        add_test(NAME ${test}_np${MFEM_MPI_NP}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
          ${MPIEXEC_PREFLAGS}
          $<TARGET_FILE:${test}> -no-vis
          ${MPIEXEC_POSTFLAGS})
      endforeach()
    endif()
  endif()
endif()
