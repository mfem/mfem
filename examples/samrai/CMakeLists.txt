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

enable_language(Fortran)

set(SAMRAI_EXAMPLES_SRCS)
list(APPEND SAMRAI_EXAMPLES_SRCS
  ex16.cpp
  )

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_samrai" target, see below.
add_custom_target(test_samrai
  ${CMAKE_CTEST_COMMAND} -R samrai USES_TERMINAL)

# Add one executable per cpp file, adding "samrai_" as prefix so the CMake
# target is unique from those in the non-SAMRAI examples. Also sets
# "test_samrai" as a target that depends on the given SAMRAI examples.
set(PFX samrai_)
add_mfem_examples(SAMRAI_EXAMPLES_SRCS ${PFX} "" test_samrai)

# Copy SAMRAI input files to build directory
configure_file(samrai_input.2d samrai_input.2d COPYONLY)
configure_file(samrai_input.3d samrai_input.3d COPYONLY)

# Remove "samrai_" prefix from executable name for consistency with GNU build
# system.
foreach(SRC_FILE ${SAMRAI_EXAMPLES_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TARGET_NAME "${PFX}${SRC_FILENAME}")
  string(REPLACE ${PFX} "" EXE_NAME ${TARGET_NAME})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXE_NAME})
endforeach()

# Add LinAdv.cpp as additional source to the ex16 target
target_sources(samrai_ex16 PRIVATE LinAdv.cpp fortran/flux2d.f fortran/flux3d.f
  fortran/fluxcorner3d.f fortran/grad2d.f fortran/grad3d.f fortran/init2d.f
  fortran/init3d.f fortran/stable2d.f fortran/stable3d.f fortran/stufprobc.f
  fortran/trace1d.f fortran/trace2d.f fortran/trace3d.f MeshOps.cpp
)

# Testing.
# The SAMRAI tests can be run separately using the target "test_samrai"
# which builds the examples and runs:
#   ctest -R samrai

if (MFEM_ENABLE_TESTING)
  # Command line options for the tests.
  # Example 16: test MFEM-SAMRAI coupled simulation
  set(EX16_2D_TEST_OPTS -m ../../data/inline-quad.mesh -i samrai_input.2d -no-vis -tf 1.0)
  set(EX16_3D_TEST_OPTS -m ../../data/star.mesh -i samrai_input.3d -no-vis -tf 1.0)

  # Add the tests
  add_test(NAME ${PFX}ex16_2d_ser
    COMMAND ${PFX}ex16 ${EX16_2D_TEST_OPTS})

  add_test(NAME ${PFX}ex16_3d_ser
    COMMAND ${PFX}ex16 ${EX16_3D_TEST_OPTS})
endif(MFEM_ENABLE_TESTING)
