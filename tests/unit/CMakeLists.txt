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

project(mfem-unit-tests NONE)

# Include the source directory for the unit tests - catch.hpp is there.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

# The following list can be updated using (in bash):
#    for d in dfem general linalg mesh fem enzyme; do ls -1 $d/*.cpp; done
set(UNIT_TESTS_SRCS
  dfem/test_diffusion.cpp
  dfem/test_divergence.cpp
  dfem/test_mass.cpp
  dfem/test_lvector_interface.cpp
  general/test_array.cpp
  general/test_scan.cpp
  general/test_arrays_by_name.cpp
  general/test_error.cpp
  general/test_mem.cpp
  general/test_reduction.cpp
  general/test_text.cpp
  general/test_umpire_mem.cpp
  general/test_zlib.cpp
  linalg/test_cg_indefinite.cpp
  linalg/test_chebyshev.cpp
  linalg/test_complex_dense_matrix.cpp
  linalg/test_complex_operator.cpp
  linalg/test_constrainedsolver.cpp
  linalg/test_direct_solvers.cpp
  linalg/test_hypre_ilu.cpp
  linalg/test_hypre_prec.cpp
  linalg/test_hypre_vector.cpp
  linalg/test_ilu.cpp
  linalg/test_matrix_block.cpp
  linalg/test_matrix_dense.cpp
  linalg/test_matrix_hypre.cpp
  linalg/test_matrix_rectangular.cpp
  linalg/test_matrix_sparse.cpp
  linalg/test_matrix_square.cpp
  linalg/test_mma.cpp
  linalg/test_ode.cpp
  linalg/test_ode2.cpp
  linalg/test_operator.cpp
  linalg/test_vector.cpp
  mesh/mesh_test_utils.cpp
  mesh/test_exodus_reader.cpp
  mesh/test_exodus_writer.cpp
  mesh/test_face_orientations.cpp
  mesh/test_fms.cpp
  mesh/test_geometric_factors.cpp
  mesh/test_mesh.cpp
  mesh/test_ncmesh.cpp
  mesh/test_nurbs.cpp
  mesh/test_periodic_mesh.cpp
  mesh/test_pmesh.cpp
  mesh/test_psubmesh.cpp
  mesh/test_submesh.cpp
  mesh/test_vtu.cpp
  mesh/test_nurbs.cpp
  mesh/test_exodus_writer.cpp
  fem/make_permuted_mesh.cpp
  fem/test_1d_bilininteg.cpp
  fem/test_2d_bilininteg.cpp
  fem/test_3d_bilininteg.cpp
  fem/test_assemblediagonalpa.cpp
  fem/test_assembly_levels.cpp
  fem/test_bilinearform.cpp
  fem/test_block_operators.cpp
  fem/test_blocknonlinearform.cpp
  fem/test_build_dof_to_arrays.cpp
  fem/test_calccurlshape.cpp
  fem/test_calcdivshape.cpp
  fem/test_calcdshape.cpp
  fem/test_calcshape.cpp
  fem/test_calcvshape.cpp
  fem/test_coefficient.cpp
  fem/test_col_lag_der.cpp
  fem/test_datacollection.cpp
  fem/test_derefine.cpp
  fem/test_dgmassinv.cpp
  fem/test_doftrans.cpp
  fem/test_domain_int.cpp
  fem/test_eigs.cpp
  fem/test_estimator.cpp
  fem/test_fa_determinism.cpp
  fem/test_face_elem_trans.cpp
  fem/test_face_permutation.cpp
  fem/test_face_restriction.cpp
  fem/test_fe_compatibility.cpp
  fem/test_fe_fixed.cpp
  fem/test_fe_pos.cpp
  fem/test_fe_symmetry.cpp
  fem/test_fe.cpp
  fem/test_get_value.cpp
  fem/test_getderivative.cpp
  fem/test_getgradient.cpp
  fem/test_getgradients.cpp
  fem/test_gslib.cpp
  fem/test_hp_transfer.cpp
  fem/test_intrules.cpp
  fem/test_intruletypes.cpp
  fem/test_inversetransform.cpp
  fem/test_lexicographic_ordering.cpp
  fem/test_lin_interp.cpp
  fem/test_linear_fes.cpp
  fem/test_linearform_ext.cpp
  fem/test_lor_batched.cpp
  fem/test_lor_dg.cpp
  fem/test_lor.cpp
  fem/test_nonlinearform.cpp
  fem/test_operatorjacobismoother.cpp
  fem/test_oscillation.cpp
  fem/test_pa_coeff.cpp
  fem/test_pa_grad.cpp
  fem/test_pa_idinterp.cpp
  fem/test_pa_kernels.cpp
  fem/test_pgridfunc_save_serial.cpp
  fem/test_poly1d.cpp
  fem/test_project_bdr_par.cpp
  fem/test_project_bdr.cpp
  fem/test_quadf_coef.cpp
  fem/test_quadinterpolator.cpp
  fem/test_quadraturefunc.cpp
  fem/test_r1d_bilininteg.cpp
  fem/test_r2d_bilininteg.cpp
  fem/test_sparse_matrix.cpp
  fem/test_sum_bilin.cpp
  fem/test_surf_blf.cpp
  fem/test_tet_reorder.cpp
  fem/test_transfer.cpp
  fem/test_var_order.cpp
  fem/test_white_noise.cpp
  enzyme/compatibility.cpp
  # The following are tested separately (keep the comment as a reminder).
  # This list can be updated using (in bash):
  #    for d in miniapps ceed; do ls -1 $d/*.cpp; done
  # miniapps/test_debug_device.cpp
  # miniapps/test_sedov.cpp
  # miniapps/test_tmop_pa.cpp
  # ceed/test_ceed.cpp
  # ceed/test_ceed_main.cpp
)

#-----------------------------------------------------------
# SERIAL CPU TESTS: unit_tests
#-----------------------------------------------------------
if (MFEM_USE_CUDA)
    set_property(SOURCE unit_test_main.cpp ${UNIT_TESTS_SRCS}
                PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
    set_property(SOURCE unit_test_main.cpp ${UNIT_TESTS_SRCS}
               PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

# All serial non-device unit tests are built into a single executable,
# 'unit_tests'.
mfem_add_executable(unit_tests unit_test_main.cpp ${UNIT_TESTS_SRCS})
target_link_libraries(unit_tests mfem)
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} unit_tests)
# Unit tests need the ../../data directory.
add_dependencies(unit_tests copy_data)
# ParSubMesh tests need meshes in ../../miniapps/multidomain
add_dependencies(unit_tests copy_miniapps_multidomain_data)
# NURBS tests need meshes in ../../miniapps/nurbs
add_dependencies(unit_tests copy_miniapps_nurbs_data)

# Copy data to the build directory.
add_custom_command(TARGET unit_tests POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data data
        COMMENT "Copying the unit tests data directory ...")

# Create a test called 'unit_tests' that runs the 'unit_tests' executable.
# The unit tests can be built and run separately from the rest of the tests:
#   make unit_tests
#   ctest -R unit_tests [-V]
if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
    add_test(NAME unit_tests COMMAND unit_tests)
endif()

#-----------------------------------------------------------
# SERIAL CUDA TESTS: gpu_unit_tests
#-----------------------------------------------------------
# Create CUDA executable and test
if (MFEM_USE_CUDA)
    # gpu_unit_tests
    set(GPU_UNIT_TESTS_SRCS gpu_unit_test_main.cpp)
    set_property(SOURCE ${GPU_UNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    mfem_add_executable(gpu_unit_tests ${GPU_UNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
    target_link_libraries(gpu_unit_tests mfem)
    add_dependencies(gpu_unit_tests copy_data)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} gpu_unit_tests)
    if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
        add_test(NAME gpu_unit_tests COMMAND gpu_unit_tests)
    endif()
endif()

#-----------------------------------------------------------
# SERIAL HIP TESTS: gpu_unit_tests
#-----------------------------------------------------------
# Create HIP 'gpu_unit_tests' executable and test
if (MFEM_USE_HIP)
    # gpu_unit_tests
    set(GPU_UNIT_TESTS_SRCS gpu_unit_test_main.cpp)
    mfem_add_executable(gpu_unit_tests ${GPU_UNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
    target_link_libraries(gpu_unit_tests mfem)
    add_dependencies(gpu_unit_tests copy_data)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} gpu_unit_tests)
    if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
        add_test(NAME gpu_unit_tests COMMAND gpu_unit_tests)
    endif()
endif()

#-----------------------------------------------------------
# SERIAL SEDOV + TMOP TESTS:
#   sedov_tests_{cpu,debug,gpu,gpu_uvm}
#   tmop_pa_tests_{cpu,debug,gpu}
#-----------------------------------------------------------
# Function to add one device serial test from the tests/unit/miniapp directory.
# All device unit tests are built into a separate executable, in order to be
# able to change the device.
function(add_serial_miniapp_test name test_uvm)
    string(TOUPPER ${name} NAME)

    set(${NAME}_TESTS_SRCS miniapps/test_${name}.cpp)
    if (MFEM_USE_CUDA)
        set_property(SOURCE ${${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif(MFEM_USE_CUDA)

    mfem_add_executable(${name}_tests_cpu ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_cpu PUBLIC MFEM_${NAME}_DEVICE="cpu")
    target_link_libraries(${name}_tests_cpu mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cpu)
    if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
        add_test(NAME ${name}_tests_cpu COMMAND ${name}_tests_cpu)
    endif()

    mfem_add_executable(${name}_tests_debug ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_debug PUBLIC MFEM_${NAME}_DEVICE="debug")
    target_link_libraries(${name}_tests_debug mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_debug)
    if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
        add_test(NAME ${name}_tests_debug COMMAND ${name}_tests_debug)
    endif()

    if (MFEM_USE_CUDA OR MFEM_USE_HIP)
        mfem_add_executable(${name}_tests_gpu ${${NAME}_TESTS_SRCS})
        target_compile_definitions(${name}_tests_gpu PUBLIC MFEM_${NAME}_DEVICE="gpu")
        target_link_libraries(${name}_tests_gpu mfem)
        add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_gpu)
        if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
            add_test(NAME ${name}_tests_gpu COMMAND ${name}_tests_gpu)
        endif()

        if (test_uvm)
            mfem_add_executable(${name}_tests_gpu_uvm ${${NAME}_TESTS_SRCS})
            target_compile_definitions(${name}_tests_gpu_uvm PUBLIC
               MFEM_${NAME}_DEVICE="gpu:uvm")
            target_link_libraries(${name}_tests_gpu_uvm mfem)
            add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME}
               ${name}_tests_gpu_uvm)
            if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
                add_test(NAME ${name}_tests_gpu_uvm COMMAND ${name}_tests_gpu_uvm)
            endif()
        endif()
    endif()
endfunction(add_serial_miniapp_test)

add_serial_miniapp_test(sedov ON)  # UVM ON
add_serial_miniapp_test(tmop_pa OFF)  # UVM OFF
# TMOP tests need meshes in ../../miniapps/meshing
add_dependencies(tmop_pa_tests_cpu copy_miniapps_meshing_data)

#-----------------------------------------------------------
# SERIAL CEED TESTS:
#   ceed_tests, ceed_tests_cuda_{ref,shared,gen}
#-----------------------------------------------------------
# Add 'ceed_tests' executable and test; add extra tests 'ceed_test_*'
if (MFEM_USE_CEED)
    set(CEED_TESTS_SRCS
      ceed/test_ceed.cpp
      ceed/test_ceed_main.cpp)
    if (MFEM_USE_CUDA)
        set_property(SOURCE ${CEED_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif(MFEM_USE_CUDA)
    mfem_add_executable(ceed_tests ${CEED_TESTS_SRCS})
    target_link_libraries(ceed_tests mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ceed_tests)
    # Add CEED tests
    add_test(NAME ceed_tests COMMAND ceed_tests)
    if (MFEM_USE_CUDA)
        add_test(NAME ceed_tests_cuda_ref
               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/ref)
        add_test(NAME ceed_tests_cuda_shared
               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/shared)
        add_test(NAME ceed_tests_cuda_gen
               COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/gen)
    endif()
endif()

#-----------------------------------------------------------
# PARALLEL CPU AND CUDA TESTS: {p,pc}unit_tests and pgpu_unit_tests
#-----------------------------------------------------------
# Define executables and tests
if (MFEM_USE_MPI)
    # punit_tests
    if (MFEM_USE_CUDA)
        set_property(SOURCE punit_test_main.cpp PROPERTY LANGUAGE CUDA)
    endif()
    mfem_add_executable(punit_tests punit_test_main.cpp ${UNIT_TESTS_SRCS})
    target_link_libraries(punit_tests mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests)
    foreach(np 1 ${MFEM_MPI_NP})
        if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
            add_test(NAME punit_tests_np=${np}
                  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
                  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:punit_tests>
                  ${MPIEXEC_POSTFLAGS})
        endif()
    endforeach()
    if (MFEM_USE_CUDA)
        # pgpu_unit_tests
        set(PGPU_UNIT_TESTS_SRCS pgpu_unit_test_main.cpp)
        set_property(SOURCE ${PGPU_UNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
        mfem_add_executable(pgpu_unit_tests ${PGPU_UNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
        add_dependencies(pgpu_unit_tests copy_data)
        target_link_libraries(pgpu_unit_tests mfem)
        add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} pgpu_unit_tests)
        foreach(np 1 ${MFEM_MPI_NP})
            if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
                add_test(NAME pgpu_unit_tests_np=${np}
                      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
                      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:pgpu_unit_tests>
                      ${MPIEXEC_POSTFLAGS})
            endif()
        endforeach()
    endif()
    if (MFEM_USE_HIP)
        # pgpu_unit_tests
        set(PGPU_UNIT_TESTS_SRCS pgpu_unit_test_main.cpp)
        mfem_add_executable(pgpu_unit_tests ${PGPU_UNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
        add_dependencies(pgpu_unit_tests copy_data)
        target_link_libraries(pgpu_unit_tests mfem)
        add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} pgpu_unit_tests)
        foreach(np 1 ${MFEM_MPI_NP})
            if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
                add_test(NAME pgpu_unit_tests_np=${np}
                      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
                      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:pgpu_unit_tests>
                      ${MPIEXEC_POSTFLAGS})
            endif()
        endforeach()
    endif()
endif(MFEM_USE_MPI)

#-----------------------------------------------------------
# PARALLEL SEDOV + TMOP TESTS:
#   psedov_tests_{cpu,debug,gpu,gpu_uvm}
#   ptmop_pa_tests_{cpu,gpu}
#-----------------------------------------------------------
# Function to add one MPI executable for a test.
function(add_mpi_executable_test name dev)
    string(TOUPPER ${name} NAME)
    string(REPLACE "_" ":" DEV ${dev})
    mfem_add_executable(p${name}_tests_${dev} ${PAR_${NAME}_TESTS_SRCS})
    target_compile_definitions(p${name}_tests_${dev} PUBLIC MFEM_${NAME}_MPI=1)
    target_compile_definitions(p${name}_tests_${dev} PUBLIC MFEM_${NAME}_DEVICE="${DEV}")
    target_link_libraries(p${name}_tests_${dev} mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} p${name}_tests_${dev})
endfunction(add_mpi_executable_test)

# Function to add one test from the tests/unit/miniapp directory.
function(add_parallel_miniapp_test name HYPRE_MM)
    string(TOUPPER ${name} NAME)

    function(add_mpi_unit_test DEV NP)
        set(test_name p${name}_tests_${DEV})
        if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
            add_test(NAME ${test_name}_np=${NP}
                COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}>
                ${MPIEXEC_POSTFLAGS})
        endif()
    endfunction()

    set(PAR_${NAME}_TESTS_SRCS miniapps/test_${name}.cpp)
    if (MFEM_USE_CUDA)
        set_property(SOURCE ${PAR_${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif()

    set(backends cpu)
    if (HYPRE_MM)
        # psedov_tests_debug_* will return MFEM_SKIP_RETURN_VALUE when all of
        # the following conditions are met:
        # * MFEM_SEDOV_MPI is defined (true here)
        # * MFEM_DEBUG is defined
        # * MFEM_SEDOV_DEVICE is "debug" (the case added here)
        # * HypreUsingGPU() is true; this is the same as: HYPRE_USING_GPU is
        #   defined and MFEM_HYPRE_VERSION < 23100 (if the version is >= 23100,
        #   the code will switch to HYPRE running on CPU).
        # We check these conditions here to skip the "debug" backend and avoid
        # the ctest failure.
        if (NOT ((${name} STREQUAL "sedov") AND MFEM_DEBUG AND
                 (HYPRE_USING_CUDA OR HYPRE_USING_HIP) AND
                 (${MFEM_HYPRE_VERSION} LESS "23100")))
            list(APPEND backends debug)
        endif()
    endif()
    if (MFEM_USE_CUDA OR MFEM_USE_HIP)
        list(APPEND backends gpu)
        if (HYPRE_MM)
            list(APPEND backends gpu_uvm)
        endif()
    endif()

    set(MPI_NPS 1 ${MFEM_MPI_NP})
    foreach(dev ${backends})
        add_mpi_executable_test(${name} ${dev})
        foreach(np ${MPI_NPS})
            add_mpi_unit_test(${dev} ${np})
        endforeach()
    endforeach()
endfunction(add_parallel_miniapp_test)

# Additional MPI unit tests
if (MFEM_USE_MPI)
    add_parallel_miniapp_test(sedov TRUE)
    add_parallel_miniapp_test(tmop_pa FALSE)
endif(MFEM_USE_MPI)

#-----------------------------------------------------------
# SERIAL DEBUG-DEVICE TESTS:
#   debug_device_tests
#-----------------------------------------------------------
set(DEBUG_DEVICE_SRCS miniapps/test_debug_device.cpp)
if (MFEM_USE_CUDA)
    set_property(SOURCE ${DEBUG_DEVICE_SRCS} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
    set_property(SOURCE ${DEBUG_DEVICE_SRCS}
                PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()
mfem_add_executable(debug_device_tests ${DEBUG_DEVICE_SRCS})
target_link_libraries(debug_device_tests mfem)
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} debug_device_tests)
add_test(NAME debug_device_tests COMMAND debug_device_tests)
