# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Include the top mfem source directory - needed by some tests, e.g. to
# #include "general/text.hpp".
include_directories(BEFORE ${PROJECT_SOURCE_DIR})
# Include the build directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})
# Include the source directory for the unit tests - catch.hpp is there.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

set(UNIT_TESTS_SRCS
  unit_test_main.cpp
  general/test_mem.cpp
  general/test_text.cpp
  linalg/test_blockMatrix.cpp
  linalg/test_densematrix.cpp
  mesh/test_mesh.cpp
  fem/test_1d_bilininteg.cpp
  fem/test_2d_bilininteg.cpp
  fem/test_3d_bilininteg.cpp
  fem/test_calcshape.cpp
  fem/test_datacollection.cpp
  fem/test_fe.cpp
  fem/test_intrules.cpp
  fem/test_intruletypes.cpp
  fem/test_inversetransform.cpp
  fem/test_lin_interp.cpp
  fem/test_linear_fes.cpp
  fem/test_quadraturefunc.cpp
  miniapps/test_sedov.cpp
)

if (MFEM_USE_CUDA)
   foreach(file IN LISTS UNIT_TESTS_SRCS)
      set_property(SOURCE ${file} PROPERTY LANGUAGE CUDA)
   endforeach()
endif()

# All non-device unit tests are built into a single executable 'unit_tests'.
add_executable(unit_tests ${UNIT_TESTS_SRCS})
target_link_libraries(unit_tests mfem)


# All device unit tests are built in another executable,
# to be able to change the device.
set(UNIT_TESTS_SRCS_SEDOV
    unit_test_main.cpp
    miniapps/test_sedov.cpp
)
if (MFEM_USE_CUDA)
   foreach(file IN LISTS UNIT_TESTS_SRCS_SEDOV)
      set_property(SOURCE ${file} PROPERTY LANGUAGE CUDA)
   endforeach()
endif()

add_executable(unit_tests_cpu ${UNIT_TESTS_SRCS_SEDOV})
target_compile_definitions(unit_tests_cpu PUBLIC MFEM_DEV_UNIT_TESTS=1)
target_compile_definitions(unit_tests_cpu PUBLIC MFEM_DEV_UNIT="cpu")
target_link_libraries(unit_tests_cpu mfem)

add_executable(unit_tests_dbg ${UNIT_TESTS_SRCS_SEDOV})
target_compile_definitions(unit_tests_dbg PUBLIC MFEM_DEV_UNIT_TESTS=1)
target_compile_definitions(unit_tests_dbg PUBLIC MFEM_DEV_UNIT="debug")
target_link_libraries(unit_tests_dbg mfem)

if (MFEM_USE_CUDA)
   # Pure CUDA test
   add_executable(unit_tests_cuda ${UNIT_TESTS_SRCS_SEDOV})
   target_compile_definitions(unit_tests_cuda PUBLIC MFEM_DEV_UNIT_TESTS=1)
   target_compile_definitions(unit_tests_cuda PUBLIC MFEM_DEV_UNIT="cuda")
   target_link_libraries(unit_tests_cuda mfem)
   # CUDA + UVM test
   add_executable(unit_tests_cuda_uvm ${UNIT_TESTS_SRCS_SEDOV})
   target_compile_definitions(unit_tests_cuda_uvm PUBLIC MFEM_DEV_UNIT_TESTS=1)
   target_compile_definitions(unit_tests_cuda_uvm PUBLIC MFEM_DEV_UNIT="cuda,uvm")
   target_link_libraries(unit_tests_cuda_uvm mfem)
   # CUDA + debug test
   add_executable(unit_tests_cuda_dbg ${UNIT_TESTS_SRCS_SEDOV})
   target_compile_definitions(unit_tests_cuda_dbg PUBLIC MFEM_DEV_UNIT_TESTS=1)
   target_compile_definitions(unit_tests_cuda_dbg PUBLIC MFEM_DEV_UNIT="cuda,debug")
   target_link_libraries(unit_tests_cuda_dbg mfem)
endif()
   
# Copy data to the build directory.
add_custom_command(TARGET unit_tests POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data data
        COMMENT "Copying the unit tests data directory ...")

# Add unit tests
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} unit_tests unit_tests_cpu unit_tests_dbg)
if (MFEM_USE_CUDA)
   add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} unit_tests_cuda unit_tests_cuda_uvm unit_tests_cuda_dbg)
endif()
   
# Create a test called 'unit_tests' that runs the 'unit_tests' executable.
# The unit tests can be built and run separately from the rest of the tests:
#   make unit_tests
#   ctest -R unit_tests [-V]
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME unit_tests_cpu COMMAND unit_tests_cpu)
#add_test(NAME unit_tests_dbg COMMAND unit_tests_dbg)

# Additional CUDA unit tests
if (MFEM_USE_CUDA)
    foreach(dev cuda cuda_dbg cuda_uvm)
        add_test(NAME unit_tests_${dev} COMMAND unit_tests_${dev})
    endforeach()
endif()

# Additional MPI unit tests
if (MFEM_USE_MPI)
   function(add_mpi_unit_test NAME NP)
      set(test_name unit_tests_${NAME})
      add_test(NAME ${test_name}_np${NP}
               COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NP} ${test_name})
   endfunction()
   set(MPI_NPS 1 2 3)   
   foreach(np ${MPI_NPS})
      add_mpi_unit_test(cpu ${np})
   endforeach()
   if (MFEM_USE_CUDA)
      foreach(dev cuda cuda_dbg cuda_uvm)
         foreach(np ${MPI_NPS})
            add_mpi_unit_test(${dev} ${np})
         endforeach()
      endforeach()
   endif()
endif()
