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
add_subdirectory(lib)

  set(NAVIER_DIR ${CMAKE_SOURCE_DIR}/miniapps/navier)
  set(NAVIER_EXTRA_INCLUDE_DIRECTORIES
    ${NAVIER_DIR}
    ${CMAKE_SOURCE_DIR}/miniapps/common
    ${CMAKE_SOURCE_DIR}/miniapps/hooke/kernels)

  add_library(navier
    ${NAVIER_DIR}/lib/navier_solver.cpp
    ${NAVIER_DIR}/kernels/mean_evaluator.cpp
    ${NAVIER_DIR}/kernels/curl_evaluator.cpp
    ${NAVIER_DIR}/kernels/stress_integrator.cpp
    ${NAVIER_DIR}/kernels/shear_stress_evaluator.cpp
    ${CMAKE_SOURCE_DIR}/miniapps/common/dist_solver.cpp)

  target_include_directories(navier PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${MFEM_INCLUDE_DIRS}
    ${TPL_INCLUDE_DIRS}
    ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  target_link_libraries(navier
    mfem
    ${TPL_LIBRARIES})

  add_mfem_miniapp(navier_mms
    MAIN examples/navier_mms.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_mms
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_mms2
    MAIN examples/navier_mms2.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_mms2
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_kovasznay
    MAIN examples/navier_kovasznay.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_kovasznay
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_kovasznay_vs
    MAIN examples/navier_kovasznay_vs.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_kovasznay_vs
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_tgv
    MAIN examples/navier_tgv.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_tgv
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_shear
    MAIN examples/navier_shear.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_shear
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_3dfoc
    MAIN examples/navier_3dfoc.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

 target_include_directories(navier_3dfoc
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_turbchan
    MAIN examples/navier_turbchan.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_turbchan
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  add_mfem_miniapp(navier_tgv2d
    MAIN examples/navier_tgv2d.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem navier)

  target_include_directories(navier_tgv2d
    PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})

  if (MFEM_USE_GSLIB)
    add_mfem_miniapp(navier_cht
      MAIN examples/navier_cht.cpp
      ${NAVIER_COMMON_FILES}
      LIBRARIES mfem navier)

    target_include_directories(navier_cht
      PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})
  
    add_mfem_miniapp(navier_fsi
      MAIN examples/navier_fsi.cpp
      ${NAVIER_COMMON_FILES}
      LIBRARIES mfem navier)

    target_include_directories(navier_fsi
      PRIVATE ${NAVIER_EXTRA_INCLUDE_DIRECTORIES})  

    LIST(APPEND NAVIER_MESH_FILES
        examples/fluid-cht.mesh
        examples/solid-cht.mesh)
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
endif () #MFEM_USE_MPI
