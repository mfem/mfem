# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
  general/test_mem.cpp
  general/test_text.cpp
  general/test_zlib.cpp
  linalg/test_complex_operator.cpp
  linalg/test_ilu.cpp
  linalg/test_matrix_block.cpp
  linalg/test_matrix_dense.cpp
  linalg/test_matrix_rectangular.cpp
  linalg/test_matrix_square.cpp
  linalg/test_ode.cpp
  linalg/test_ode2.cpp
  linalg/test_operator.cpp
  linalg/test_cg_indefinite.cpp
  mesh/test_mesh.cpp
  fem/test_1d_bilininteg.cpp
  fem/test_2d_bilininteg.cpp
  fem/test_3d_bilininteg.cpp
  fem/test_assemblediagonalpa.cpp
  fem/test_calcshape.cpp
  fem/test_datacollection.cpp
  fem/test_face_permutation.cpp
  fem/test_fe.cpp
  fem/test_intrules.cpp
  fem/test_intruletypes.cpp
  fem/test_inversetransform.cpp
  fem/test_lin_interp.cpp
  fem/test_linear_fes.cpp
  fem/test_operatorjacobismoother.cpp
  fem/test_pa_coeff.cpp
  fem/test_pa_kernels.cpp
  fem/test_quadf_coef.cpp
  fem/test_quadraturefunc.cpp
  miniapps/test_sedov.cpp
  miniapps/test_tmop.cpp
)

if (MFEM_USE_CUDA)
   set_property(SOURCE ${UNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
endif()

# All serial non-device unit tests are built into a single executable 'unit_tests'.
add_executable(unit_tests unit_test_main.cpp ${UNIT_TESTS_SRCS})
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

# Copy blade.mesh to the tests/unit build directory.
configure_file("${PROJECT_SOURCE_DIR}/miniapps/meshing/blade.mesh"
               "${PROJECT_BINARY_DIR}/tests/unit/blade.mesh")

# Copy star.mesh to the tests/unit build directory.
configure_file("${PROJECT_SOURCE_DIR}/data/star.mesh"
               "${PROJECT_BINARY_DIR}/tests/unit/star.mesh")
               
# Copy toroid-hex.mesh to the tests/unit build directory.
configure_file("${PROJECT_SOURCE_DIR}/data/toroid-hex.mesh"
               "${PROJECT_BINARY_DIR}/tests/unit/toroid-hex.mesh")

# Add unit tests
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME}
                unit_tests
                sedov_tests_cpu sedov_tests_debug
                tmop_tests_cpu tmop_tests_debug)

# Create a test called 'punit_tests' that runs the 'punit_tests' executable.
if (MFEM_USE_MPI)
   add_executable(punit_tests punit_test_main.cpp ${UNIT_TESTS_SRCS})
   target_link_libraries(punit_tests mfem)
   add_test(NAME punit_tests COMMAND punit_tests)
endif(MFEM_USE_MPI)

# Function to add one test from the tests/unit/miniapp directory.
function(add_parallel_miniapp_test name)
    string(TOUPPER name NAME)

    add_test(NAME ${name}_tests_cpu COMMAND ${name}_tests_cpu)
    add_test(NAME ${name}_tests_debug COMMAND ${name}_tests_debug)

    # Additional CUDA unit tests
    if (MFEM_USE_CUDA)
       add_test(NAME ${name}_tests_cuda COMMAND ${name}_tests_cuda)
       add_test(NAME ${name}_tests_cuda_uvm COMMAND ${name}_tests_cuda_uvm)
    endif()

    # Additional MPI tests
    if (MFEM_USE_MPI)
       set(PAR_${NAME}_TESTS_SRCS punit_test_main.cpp miniapps/test_${name}.cpp)
       if (MFEM_USE_CUDA)
          set_property(SOURCE ${PAR_${NAME}_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
       endif()

       add_executable(p${name}_tests_cpu ${PAR_${NAME}_TESTS_SRCS})
       target_compile_definitions(p${name}_tests_cpu PUBLIC MFEM_${NAME}_MPI=1)
       target_compile_definitions(p${name}_tests_cpu PUBLIC MFEM_${NAME}_TESTS=1)
       target_compile_definitions(p${name}_tests_cpu PUBLIC MFEM_${NAME}_DEVICE="cpu")
       target_link_libraries(p${name}_tests_cpu mfem)

       add_executable(p${name}_tests_debug ${PAR_${NAME}_TESTS_SRCS})
       target_compile_definitions(p${name}_tests_debug PUBLIC MFEM_${NAME}_MPI=1)
       target_compile_definitions(p${name}_tests_debug PUBLIC MFEM_${NAME}_TESTS=1)
       target_compile_definitions(p${name}_tests_debug PUBLIC MFEM_${NAME}_DEVICE="debug")
       target_link_libraries(p${name}_tests_debug mfem)

       if (MFEM_USE_CUDA)
          add_executable(p${name}_tests_cuda ${PAR_${NAME}_TESTS_SRCS})
          target_compile_definitions(p${name}_tests_cuda PUBLIC MFEM_${NAME}_MPI=1)
          target_compile_definitions(p${name}_tests_cuda PUBLIC MFEM_${NAME}_TESTS=1)
          target_compile_definitions(p${name}_tests_cuda PUBLIC MFEM_${NAME}_DEVICE="cuda")
          target_link_libraries(p${name}_tests_cuda mfem)

          add_executable(p${name}_tests_cuda_uvm ${PAR_${NAME}_TESTS_SRCS})
          target_compile_definitions(p${name}_tests_cuda_uvm PUBLIC MFEM_${NAME}_MPI=1)
          target_compile_definitions(p${name}_tests_cuda_uvm PUBLIC MFEM_${NAME}_TESTS=1)
          target_compile_definitions(p${name}_tests_cuda_uvm PUBLIC MFEM_${NAME}_DEVICE="cuda:uvm")
          target_link_libraries(p${name}_tests_cuda_uvm mfem)
       endif()

       add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests p${name}_tests_cpu p${name}_tests_debug)
       if (MFEM_USE_CUDA)
          add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} p${name}_tests_cuda)
          add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} p${name}_tests_cuda_uvm)
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
          add_mpi_unit_test(debug ${np})
       endforeach()
       if (MFEM_USE_CUDA)
          foreach(dev cuda cuda_uvm)
             foreach(np ${MPI_NPS})
                add_mpi_unit_test(${dev} ${np})
             endforeach()
          endforeach()
       endif()
    endif(MFEM_USE_MPI)
endfunction(add_parallel_miniapp_test)

# Additional MPI unit tests
add_parallel_miniapp_test(sedov)
add_parallel_miniapp_test(tmop)
