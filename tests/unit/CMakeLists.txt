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
  miniapps/test_sedov.cpp
  miniapps/test_tmop.cpp
)

if (MFEM_USE_CUDA)
   set_property(SOURCE ${UNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
endif()
if (MFEM_USE_HIP)
  set_property(SOURCE unit_test_main.cpp ${UNIT_TESTS_SRCS} PROPERTY HIP_SOURCE_PROPERTY_FORMAT TRUE)
endif()

# All serial non-device unit tests are built into a single executable 'unit_tests'.
mfem_add_executable(unit_tests unit_test_main.cpp ${UNIT_TESTS_SRCS})
# Unit tests need the ../../data directory.
add_dependencies(unit_tests copy_data)
target_link_libraries(unit_tests mfem)

# Function to add one device serial test from the tests/unit/miniapp directory.
# All device unit tests are built into a seperate executable, in order to be
# able to change the device.
function(add_serial_miniapp_test name)
    string(TOUPPER name NAME)

    set(${NAME}_TESTS_SRCS
        unit_test_main.cpp
        miniapps/test_${name}.cpp)

    if (MFEM_USE_CUDA)
       set_property(SOURCE ${${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
    endif(MFEM_USE_CUDA)

    add_executable(${name}_tests_cpu ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_cpu PUBLIC MFEM_${NAME}_TESTS=1)
    target_compile_definitions(${name}_tests_cpu PUBLIC MFEM_${NAME}_DEVICE="cpu")
    target_link_libraries(${name}_tests_cpu mfem)

    add_executable(${name}_tests_debug ${${NAME}_TESTS_SRCS})
    target_compile_definitions(${name}_tests_debug PUBLIC MFEM_${NAME}_TESTS=1)
    target_compile_definitions(${name}_tests_debug PUBLIC MFEM_${NAME}_DEVICE="debug")
    target_link_libraries(${name}_tests_debug mfem)

    if (MFEM_USE_CUDA)
       add_executable(${name}_tests_cuda ${${NAME}_TESTS_SRCS})
       target_compile_definitions(${name}_tests_cuda PUBLIC MFEM_${NAME}_TESTS=1)
       target_compile_definitions(${name}_tests_cuda PUBLIC MFEM_${NAME}_DEVICE="cuda")
       target_link_libraries(${name}_tests_cuda mfem)

       add_executable(${name}_tests_cuda_uvm ${${NAME}_TESTS_SRCS})
       target_compile_definitions(${name}_tests_cuda_uvm PUBLIC MFEM_${NAME}_TESTS=1)
       target_compile_definitions(${name}_tests_cuda_uvm PUBLIC MFEM_${NAME}_DEVICE="cuda:uvm")
       target_link_libraries(${name}_tests_cuda_uvm mfem)

       add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cuda)
       add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ${name}_tests_cuda_uvm)
    endif(MFEM_USE_CUDA)
endfunction(add_serial_miniapp_test)

add_serial_miniapp_test(sedov)
add_serial_miniapp_test(tmop)

# Create CUDA cunit_tests executable
if (MFEM_USE_CUDA)
  set(CUNIT_TESTS_SRCS cunit_test_main.cpp)
  set_property(SOURCE ${CUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
  add_executable(cunit_tests ${CUNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
  add_dependencies(cunit_tests copy_data)
  target_link_libraries(cunit_tests mfem)
  add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} cunit_tests)
endif()

# CEED tests sources
if (MFEM_USE_CEED)
   set(CEED_TESTS_SRCS
      ceed/test_ceed.cpp
      ceed/test_ceed_main.cpp)
   add_executable(ceed_tests ${CEED_TESTS_SRCS})
   target_link_libraries(ceed_tests mfem)
endif()

# Copy data to the build directory.
add_custom_command(TARGET unit_tests POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data data
        COMMENT "Copying the unit tests data directory ...")

# Add unit tests
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} unit_tests sedov_tests_cpu sedov_tests_debug)
if (MFEM_USE_CUDA)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} sedov_tests_cuda)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} sedov_tests_cuda_uvm)
endif()

if (MFEM_USE_CEED)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} ceed_tests)
endif()

# Create a test called 'unit_tests' that runs the 'unit_tests' executable.
# The unit tests can be built and run separately from the rest of the tests:
#   make unit_tests
#   ctest -R unit_tests [-V]
add_test(NAME unit_tests COMMAND unit_tests)

# Copy data to the build directory.
add_custom_command(TARGET unit_tests POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/data data
                   COMMENT "Copying the unit tests data directory ...")

# Additional CUDA unit tests
if (MFEM_USE_CUDA)
   add_test(NAME cunit_tests COMMAND cunit_tests)
   add_test(NAME sedov_tests_cuda COMMAND sedov_tests_cuda)
   add_test(NAME sedov_tests_cuda_uvm COMMAND sedov_tests_cuda_uvm)
endif()

# Add unit tests
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME}
                unit_tests
                sedov_tests_cpu sedov_tests_debug
                tmop_tests_cpu tmop_tests_debug)

# Add CEED tests
if (MFEM_USE_CEED)
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

# Create a test called 'punit_tests' that runs the 'punit_tests' executable.
# Additional MPI unit tests
if (MFEM_USE_MPI)
   add_executable(punit_tests punit_test_main.cpp ${UNIT_TESTS_SRCS})
   target_link_libraries(punit_tests mfem)
   add_test(NAME punit_tests COMMAND punit_tests)
   add_test(NAME punit_tests_np=${MFEM_MPI_NP}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:punit_tests>
            ${MPIEXEC_POSTFLAGS})
endif(MFEM_USE_MPI)

# Function to add one MPI executable for a test.
function(add_mpi_executable_test name dev)
    string(TOUPPER name NAME)
    string(REPLACE "_" ":" DEV name)
    add_executable(p${name}_tests_${dev} ${PAR_${NAME}_TESTS_SRCS})
    target_compile_definitions(p${name}_tests_${dev} PUBLIC MFEM_${NAME}_MPI=1)
    target_compile_definitions(p${name}_tests_${dev} PUBLIC MFEM_${NAME}_TESTS=1)
    target_compile_definitions(p${name}_tests_${dev} PUBLIC MFEM_${NAME}_DEVICE="${DEV}")
    target_link_libraries(p${name}_tests_${dev} mfem)
endfunction(add_mpi_executable_test)

# Function to add one test from the tests/unit/miniapp directory.
function(add_parallel_miniapp_test name HYPRE_MM)
    string(TOUPPER name NAME)

    add_test(NAME ${name}_tests_cpu COMMAND ${name}_tests_cpu)
    add_test(NAME ${name}_tests_debug COMMAND ${name}_tests_debug)

    # Additional CUDA unit tests
   if (MFEM_USE_CUDA)
      add_test(NAME ${name}_tests_cuda COMMAND ${name}_tests_cuda)
#     add_executable(psedov_tests_cuda ${PAR_SEDOV_TESTS_SRCS})
#     target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_MPI=1)
#     target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_TESTS=1)
#     target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_DEVICE="cuda")
#     target_link_libraries(psedov_tests_cuda mfem)

      add_test(NAME ${name}_tests_cuda_uvm COMMAND ${name}_tests_cuda_uvm)
#     add_executable(psedov_tests_cuda_uvm ${PAR_SEDOV_TESTS_SRCS})
#     target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_MPI=1)
#     target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_TESTS=1)
#     target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_DEVICE="cuda:uvm")
#     target_link_libraries(psedov_tests_cuda_uvm mfem)

      set(PCUNIT_TESTS_SRCS pcunit_test_main.cpp)
      set_property(SOURCE ${PCUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
      add_executable(pcunit_tests ${PCUNIT_TESTS_SRCS})
      add_dependencies(pcunit_tests copy_data)
      target_link_libraries(pcunit_tests mfem)
      add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} pcunit_tests)
   endif()

    # Additional MPI tests
    if (MFEM_USE_MPI)
       set(PAR_${NAME}_TESTS_SRCS punit_test_main.cpp miniapps/test_${name}.cpp)
       if (MFEM_USE_CUDA)
          set_property(SOURCE ${PAR_${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
       endif()

       add_mpi_executable_test(${name} cpu)
       if (HYPRE_MM)
           add_mpi_executable_test(${name} debug)
       endif()

       if (MFEM_USE_CUDA)
          add_mpi_executable_test(${name} cuda)
          if (HYPRE_MM)
              add_mpi_executable_test(${name} cuda_uvm)
          endif()
       endif()

       if (NOT HYPRE_MM)
          add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests p${name}_tests_cpu)
       else()
          add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests p${name}_tests_cpu p${name}_tests_debug)
       endif()

       if (MFEM_USE_CUDA)
          add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} p${name}_tests_cuda)
          if (HYPRE_MM)
             add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} p${name}_tests_cuda_uvm)
          endif()
       endif()

       function(add_mpi_unit_test NAME NP)
          set(test_name p${name}_tests_${NAME})
          add_test(NAME ${test_name}_np=${NP}
                   COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP}
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${test_name}>
                   ${MPIEXEC_POSTFLAGS})
       endfunction()
       set(MPI_NPS 1 ${MFEM_MPI_NP})
       foreach(np ${MPI_NPS})
          add_mpi_unit_test(cpu ${np})
          if (HYPRE_MM)
            add_mpi_unit_test(debug ${np})
          endif()
       endforeach()
       if (MFEM_USE_CUDA)
          set(backends cuda)
          if (HYPRE_MM)
             list(APPEND backends cuda_uvm)
          endif()
          foreach(dev ${backends})
             foreach(np ${MPI_NPS})
                add_mpi_unit_test(${dev} ${np})
             endforeach()
          endforeach()
       endif()
    endif(MFEM_USE_MPI)
endfunction(add_parallel_miniapp_test)

# Additional MPI unit tests
add_parallel_miniapp_test(sedov TRUE)
add_parallel_miniapp_test(tmop FALSE)
