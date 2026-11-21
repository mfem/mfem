# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
  # The Navier solver and its kernels are built once as a library, which is
  # also used by the FSI miniapps (miniapps/fsi).
  set(NAVIER_LIB_SOURCES
      navier_solver.cpp
      kernels/curl_evaluator.cpp
      kernels/mean_evaluator.cpp
      kernels/shear_stress_evaluator.cpp
      kernels/shear_stress_integrator.cpp
      kernels/vector_boundary_AMP_LFIntegrator.cpp)
  set(NAVIER_LIB_HEADERS
      navier_solver.hpp
      kernels/boundary_normal_pressure_integrator.hpp
      kernels/boundary_normal_stress_evaluator.hpp
      kernels/boundary_normal_stress_integrator.hpp
      kernels/curl_evaluator.hpp
      kernels/custom_navier_coefficients.hpp
      kernels/mean_evaluator.hpp
      kernels/shear_stress_evaluator.hpp
      kernels/shear_stress_evaluator_kernels.hpp
      kernels/shear_stress_integrator.hpp
      kernels/vector_boundary_AMP_LFIntegrator.hpp)

  if(MFEM_USE_GSLIB)
    list(APPEND NAVIER_LIB_SOURCES
      navier_particles.cpp)
    list(APPEND NAVIER_LIB_HEADERS
      navier_particles.hpp)
  endif()

  convert_filenames_to_full_paths(NAVIER_LIB_SOURCES)
  convert_filenames_to_full_paths(NAVIER_LIB_HEADERS)

  if (MFEM_USE_CUDA)
    set_property(SOURCE ${NAVIER_LIB_SOURCES} PROPERTY LANGUAGE CUDA)
  endif()
  if (MFEM_USE_HIP)
    set_property(SOURCE ${NAVIER_LIB_SOURCES}
      PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
  endif()

  mfem_add_library(navier ${NAVIER_LIB_SOURCES} ${NAVIER_LIB_HEADERS})
  target_include_directories(navier PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/miniapps/common-fsi
    ${PROJECT_SOURCE_DIR}/miniapps/hooke/kernels)
  target_link_libraries(navier mfem-common-fsi mfem)

  add_mfem_miniapp(navier_mms
    MAIN navier_mms.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_kovasznay
    MAIN navier_kovasznay.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_kovasznay_vs
    MAIN navier_kovasznay_vs.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_tgv
    MAIN navier_tgv.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_shear
    MAIN navier_shear.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_3dfoc
    MAIN navier_3dfoc.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_turbchan
    MAIN navier_turbchan.cpp
    LIBRARIES navier)

  add_mfem_miniapp(navier_test_tangential
    MAIN kernels/test_custom_navier_tangential.cpp
    LIBRARIES navier)

  set(MESH_FILE "channel-bifurcation-2d.mesh")
  add_custom_command(OUTPUT data_is_copied
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/data/${MESH_FILE}" .
    COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
    COMMENT "Copying navier miniapps data files ...")
  add_custom_target(copy_miniapps_navier_data DEPENDS data_is_copied)

  add_mfem_miniapp(navier_bifurcation
    MAIN navier_bifurcation.cpp
    LIBRARIES navier mfem-common)
add_dependencies(navier_bifurcation copy_miniapps_navier_data)

  if (MFEM_USE_GSLIB)
    add_mfem_miniapp(navier_cht
      MAIN navier_cht.cpp
      LIBRARIES navier)

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
