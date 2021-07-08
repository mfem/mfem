# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Include the top mfem source directory - needed by some tests, e.g. to
# #include "general/text.hpp".
include_directories(BEFORE ${PROJECT_SOURCE_DIR})
# Include the build directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})
# Include the source directory for the unit tests - catch.hpp is there.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set(UNIT_TESTS_SRCS
  general/test_array.cpp
  general/test_mem.cpp
  general/test_text.cpp
  general/test_umpire_mem.cpp
  general/test_zlib.cpp
  linalg/test_cg_indefinite.cpp
  linalg/test_chebyshev.cpp
  linalg/test_complex_operator.cpp
  linalg/test_constrainedsolver.cpp
  linalg/test_direct_solvers.cpp
  linalg/test_hypre_ilu.cpp
  linalg/test_ilu.cpp
  linalg/test_matrix_block.cpp
  linalg/test_matrix_dense.cpp
  linalg/test_matrix_hypre.cpp
  linalg/test_matrix_rectangular.cpp
  linalg/test_matrix_sparse.cpp
  linalg/test_matrix_square.cpp
  linalg/test_ode.cpp
  linalg/test_ode2.cpp
  linalg/test_operator.cpp
  linalg/test_vector.cpp
  mesh/test_mesh.cpp
  mesh/test_ncmesh.cpp
  mesh/test_pmesh.cpp
  mesh/test_periodic_mesh.cpp
  mesh/test_vtu.cpp
  fem/test_1d_bilininteg.cpp
  fem/test_2d_bilininteg.cpp
  fem/test_3d_bilininteg.cpp
  fem/test_assemblediagonalpa.cpp
  fem/test_assembly_levels.cpp
  fem/test_bilinearform.cpp
  fem/test_blocknonlinearform.cpp
  fem/test_calcshape.cpp
  fem/test_datacollection.cpp
  fem/test_derefine.cpp
  fem/test_estimator.cpp
  fem/test_face_elem_trans.cpp
  fem/test_face_permutation.cpp
  fem/test_fe.cpp
  fem/test_get_value.cpp
  fem/test_intrules.cpp
  fem/test_intruletypes.cpp
  fem/test_inversetransform.cpp
  fem/test_lexicographic_ordering.cpp
  fem/test_lin_interp.cpp
  fem/test_linear_fes.cpp
  fem/test_operatorjacobismoother.cpp
  fem/test_pa_coeff.cpp
  fem/test_pa_grad.cpp
  fem/test_pa_idinterp.cpp
  fem/test_pa_kernels.cpp
  fem/test_quadf_coef.cpp
  fem/test_quadraturefunc.cpp
  fem/test_sparse_matrix.cpp
  fem/test_sum_bilin.cpp
  fem/test_tet_reorder.cpp
  fem/test_transfer.cpp
  fem/test_var_order.cpp
  # The following are tested separately (keep the comment as a reminder):
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

# Copy data to the build directory.
add_custom_command(TARGET unit_tests POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data data
        COMMENT "Copying the unit tests data directory ...")

# Create a test called 'unit_tests' that runs the 'unit_tests' executable.
# The unit tests can be built and run separately from the rest of the tests:
#   make unit_tests
#   ctest -R unit_tests [-V]
add_test(NAME unit_tests COMMAND unit_tests)

#-----------------------------------------------------------
# SERIAL CUDA TESTS: cunit_tests
#-----------------------------------------------------------
# Create CUDA 'cunit_tests' executable and test
if (MFEM_USE_CUDA)
   set(CUNIT_TESTS_SRCS cunit_test_main.cpp)
   set_property(SOURCE ${CUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
   add_executable(cunit_tests ${CUNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
   target_link_libraries(cunit_tests mfem)
   add_dependencies(cunit_tests copy_data)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} cunit_tests)
   add_test(NAME cunit_tests COMMAND cunit_tests)
endif()

#-----------------------------------------------------------
# SERIAL SEDOV + TMOP TESTS:
#     sedov_tests_{cpu,debug,cuda,cuda_uvm}
#   tmop_pa_tests_{cpu,debug,cuda}
#-----------------------------------------------------------
# Function to add one device serial test from the tests/unit/miniapp directory.
# All device unit tests are built into a separate executable, in order to be
# able to change the device.
function(add_serial_miniapp_test name test_uvm)
    string(TOUPPER ${name} NAME)

    set(${NAME}_TESTS_SRCS
        unit_test_main.cpp
        miniapps/test_${name}.cpp)

    if (MFEM_USE_CUDA)
       set_property(SOURCE ${${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif(MFEM_USE_CUDA)

    add_executable(${name}_tests_cpu ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_cpu PUBLIC MFEM_${NAME}_DEVICE="cpu")
    target_link_libraries(${name}_tests_cpu mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cpu)
    add_test(NAME ${name}_tests_cpu COMMAND ${name}_tests_cpu)

    add_executable(${name}_tests_debug ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_debug PUBLIC MFEM_${NAME}_DEVICE="debug")
    target_link_libraries(${name}_tests_debug mfem)
    add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_debug)
    add_test(NAME ${name}_tests_debug COMMAND ${name}_tests_debug)

    if (MFEM_USE_CUDA)
       add_executable(${name}_tests_cuda ${${NAME}_TESTS_SRCS})
       target_compile_definitions(${name}_tests_cuda PUBLIC MFEM_${NAME}_DEVICE="cuda")
       target_link_libraries(${name}_tests_cuda mfem)
       add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cuda)
       add_test(NAME ${name}_tests_cuda COMMAND ${name}_tests_cuda)

       if (test_uvm)
           add_executable(${name}_tests_cuda_uvm ${${NAME}_TESTS_SRCS})
           target_compile_definitions(${name}_tests_cuda_uvm PUBLIC
               MFEM_${NAME}_DEVICE="cuda:uvm")
           target_link_libraries(${name}_tests_cuda_uvm mfem)
           add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME}
               ${name}_tests_cuda_uvm)
           add_test(NAME ${name}_tests_cuda_uvm COMMAND ${name}_tests_cuda_uvm)
       endif()
    endif(MFEM_USE_CUDA)
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
   add_executable(ceed_tests ${CEED_TESTS_SRCS})
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
# PARALLEL CPU AND CUDA TESTS: {p,pc}unit_tests
#-----------------------------------------------------------
# Define executables and tests 'punit_tests' and 'pcunit_tests'
if (MFEM_USE_MPI)
   if (MFEM_USE_CUDA)
      set_property(SOURCE punit_test_main.cpp PROPERTY LANGUAGE CUDA)
   endif()
   add_executable(punit_tests punit_test_main.cpp ${UNIT_TESTS_SRCS})
   target_link_libraries(punit_tests mfem)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests)
   foreach(np 1 ${MFEM_MPI_NP})
      add_test(NAME punit_tests_np=${np}
               COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
               ${MPIEXEC_PREFLAGS} $<TARGET_FILE:punit_tests>
               ${MPIEXEC_POSTFLAGS})
   endforeach()
   if (MFEM_USE_CUDA)
      set(PCUNIT_TESTS_SRCS pcunit_test_main.cpp)
      set_property(SOURCE ${PCUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
      add_executable(pcunit_tests ${PCUNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
      add_dependencies(pcunit_tests copy_data)
      target_link_libraries(pcunit_tests mfem)
      add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} pcunit_tests)
      foreach(np 1 ${MFEM_MPI_NP})
         add_test(NAME pcunit_tests_np=${np}
                  COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${np}
                  ${MPIEXEC_PREFLAGS} $<TARGET_FILE:pcunit_tests>
                  ${MPIEXEC_POSTFLAGS})
      endforeach()
   endif()
endif(MFEM_USE_MPI)

#-----------------------------------------------------------
# PARALLEL SEDOV + TMOP TESTS:
#     psedov_tests_{cpu,debug,cuda,cuda_uvm}
#   ptmop_pa_tests_{cpu,cuda}
#-----------------------------------------------------------
# Function to add one MPI executable for a test.
function(add_mpi_executable_test name dev)
    string(TOUPPER ${name} NAME)
    string(REPLACE "_" ":" DEV ${dev})
    add_executable(p${name}_tests_${dev} ${PAR_${NAME}_TESTS_SRCS})
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
        add_test(NAME ${test_name}_np=${NP}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}>
            ${MPIEXEC_POSTFLAGS})
    endfunction()

    set(PAR_${NAME}_TESTS_SRCS punit_test_main.cpp miniapps/test_${name}.cpp)
    if (MFEM_USE_CUDA)
        set_property(SOURCE ${PAR_${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif()

    set(backends cpu)
    if (HYPRE_MM)
        list(APPEND backends debug)
    endif()
    if (MFEM_USE_CUDA)
        list(APPEND backends cuda)
        if (HYPRE_MM)
            list(APPEND backends cuda_uvm)
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
