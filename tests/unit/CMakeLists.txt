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
  linalg/test_cg_indefinite.cpp
  linalg/test_vector.cpp
  mesh/test_mesh.cpp
  mesh/test_ncmesh.cpp
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
  fem/test_blocknonlinearform.cpp
  miniapps/test_sedov.cpp
)

if (MFEM_USE_CUDA)
   set_property(SOURCE ${UNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
endif()

# All serial non-device unit tests are built into a single executable 'unit_tests'.
add_executable(unit_tests unit_test_main.cpp ${UNIT_TESTS_SRCS})
# Unit tests need the ../../data directory.
add_dependencies(unit_tests copy_data)
target_link_libraries(unit_tests mfem)

if (MFEM_USE_CUDA)
  set(CUNIT_TESTS_SRCS
    cunit_test_main.cpp
  )
  set_property(SOURCE ${CUNIT_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
  add_executable(cunit_tests ${CUNIT_TESTS_SRCS} ${UNIT_TESTS_SRCS})
  add_dependencies(cunit_tests copy_data)
  target_link_libraries(cunit_tests mfem)

  add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} cunit_tests)
endif()

if (MFEM_USE_CEED)
   set(CEED_TESTS_SRCS
      ceed/test_ceed.cpp
      ceed/test_ceed_main.cpp
   )
endif()

# All device unit tests are built into another executable, in order to be able
# to change the device.
set(SEDOV_TESTS_SRCS
    unit_test_main.cpp
    miniapps/test_sedov.cpp
)

if (MFEM_USE_CUDA)
   set_property(SOURCE ${SEDOV_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
endif()

add_executable(sedov_tests_cpu ${SEDOV_TESTS_SRCS})
target_compile_definitions(sedov_tests_cpu PUBLIC MFEM_SEDOV_TESTS=1)
target_compile_definitions(sedov_tests_cpu PUBLIC MFEM_SEDOV_DEVICE="cpu")
target_link_libraries(sedov_tests_cpu mfem)

add_executable(sedov_tests_debug ${SEDOV_TESTS_SRCS})
target_compile_definitions(sedov_tests_debug PUBLIC MFEM_SEDOV_TESTS=1)
target_compile_definitions(sedov_tests_debug PUBLIC MFEM_SEDOV_DEVICE="debug")
target_link_libraries(sedov_tests_debug mfem)

if (MFEM_USE_CUDA)
   add_executable(sedov_tests_cuda ${SEDOV_TESTS_SRCS})
   target_compile_definitions(sedov_tests_cuda PUBLIC MFEM_SEDOV_TESTS=1)
   target_compile_definitions(sedov_tests_cuda PUBLIC MFEM_SEDOV_DEVICE="cuda")
   target_link_libraries(sedov_tests_cuda mfem)

   add_executable(sedov_tests_cuda_uvm ${SEDOV_TESTS_SRCS})
   target_compile_definitions(sedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_TESTS=1)
   target_compile_definitions(sedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_DEVICE="cuda:uvm")
   target_link_libraries(sedov_tests_cuda_uvm mfem)
endif()

if (MFEM_USE_CEED)
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
add_test(NAME sedov_tests_cpu COMMAND sedov_tests_cpu)
add_test(NAME sedov_tests_debug COMMAND sedov_tests_debug)

# Additional CUDA unit tests
if (MFEM_USE_CUDA)
   add_test(NAME cunit_tests COMMAND cunit_tests)
   add_test(NAME sedov_tests_cuda COMMAND sedov_tests_cuda)
   add_test(NAME sedov_tests_cuda_uvm COMMAND sedov_tests_cuda_uvm)
endif()

if (MFEM_USE_CEED)
   add_test(NAME ceed_tests COMMAND ceed_tests)
   if (MFEM_USE_CUDA)
      add_test(NAME ceed_tests_cuda_ref COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/ref)
      add_test(NAME ceed_tests_cuda_shared COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/shared)
      add_test(NAME ceed_tests_cuda_gen COMMAND ceed_tests --device ceed-cuda:/gpu/cuda/gen)
   endif()
endif()

# Additional MPI unit tests
if (MFEM_USE_MPI)
   add_executable(punit_tests punit_test_main.cpp ${UNIT_TESTS_SRCS})
   target_link_libraries(punit_tests mfem)
   add_test(NAME punit_tests COMMAND punit_tests)

   set(PAR_SEDOV_TESTS_SRCS punit_test_main.cpp miniapps/test_sedov.cpp)
   if (MFEM_USE_CUDA)
      set_property(SOURCE ${PAR_SEDOV_TESTS_SRCS} PROPERTY LANGUAGE CUDA)
   endif()

   add_executable(psedov_tests_cpu ${PAR_SEDOV_TESTS_SRCS})
   target_compile_definitions(psedov_tests_cpu PUBLIC MFEM_SEDOV_MPI=1)
   target_compile_definitions(psedov_tests_cpu PUBLIC MFEM_SEDOV_TESTS=1)
   target_compile_definitions(psedov_tests_cpu PUBLIC MFEM_SEDOV_DEVICE="cpu")
   target_link_libraries(psedov_tests_cpu mfem)

   add_executable(psedov_tests_debug ${PAR_SEDOV_TESTS_SRCS})
   target_compile_definitions(psedov_tests_debug PUBLIC MFEM_SEDOV_MPI=1)
   target_compile_definitions(psedov_tests_debug PUBLIC MFEM_SEDOV_TESTS=1)
   target_compile_definitions(psedov_tests_debug PUBLIC MFEM_SEDOV_DEVICE="debug")
   target_link_libraries(psedov_tests_debug mfem)

   if (MFEM_USE_CUDA)
      add_executable(psedov_tests_cuda ${PAR_SEDOV_TESTS_SRCS})
      target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_MPI=1)
      target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_TESTS=1)
      target_compile_definitions(psedov_tests_cuda PUBLIC MFEM_SEDOV_DEVICE="cuda")
      target_link_libraries(psedov_tests_cuda mfem)

      add_executable(psedov_tests_cuda_uvm ${PAR_SEDOV_TESTS_SRCS})
      target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_MPI=1)
      target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_TESTS=1)
      target_compile_definitions(psedov_tests_cuda_uvm PUBLIC MFEM_SEDOV_DEVICE="cuda:uvm")
      target_link_libraries(psedov_tests_cuda_uvm mfem)
   endif()

   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} punit_tests psedov_tests_cpu psedov_tests_debug)
   if (MFEM_USE_CUDA)
      add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} psedov_tests_cuda)
      add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} psedov_tests_cuda_uvm)
   endif()

   function(add_mpi_unit_test NAME NP)
      set(test_name psedov_tests_${NAME})
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
endif()
