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
  general/text-test.cpp
  linalg/test_blockMatrix.cpp
  linalg/test_complex_operator.cpp
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
  )

# All unit tests are built into a single executable 'unit_tests'.
add_executable(unit_tests ${UNIT_TESTS_SRCS})
target_link_libraries(unit_tests mfem)
add_custom_command(TARGET unit_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/data data
  COMMENT "Copying the unit tests data directory ...")
add_dependencies(${MFEM_ALL_TESTS_TARGET_NAME} unit_tests)

# Create a test called 'unit_tests' that runs the 'unit_tests' executable.
# The unit tests can be built and run separately from the rest of the tests:
#   make unit_tests
#   ctest -R unit_tests [-V]
add_test(NAME unit_tests COMMAND unit_tests)
