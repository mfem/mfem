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

set(CHECKPOINTING_EXAMPLES_SRCS)
list(APPEND CHECKPOINTING_EXAMPLES_SRCS
  ex16.cpp
  )

if (MFEM_USE_MPI)
  list(APPEND CHECKPOINTING_EXAMPLES_SRCS
    ex16p.cpp
    )
endif()

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_checkpointing" target, see below.
add_custom_target(test_checkpointing
  ${CMAKE_CTEST_COMMAND} -R checkpointing USES_TERMINAL)

# Add one executable per cpp file, adding "checkpointing_" as prefix so the
# CMake target is unique from those in the other examples. Also sets
# "test_checkpointing" as a target that depends on the given examples.
set(PFX checkpointing_)
add_mfem_examples(CHECKPOINTING_EXAMPLES_SRCS ${PFX} "" test_checkpointing)

# Remove "checkpointing_" prefix from executable name for consistency with GNU
# build system.
foreach(SRC_FILE ${CHECKPOINTING_EXAMPLES_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
  string(REPLACE ".cpp" "" TARGET_NAME "${PFX}${SRC_FILENAME}")
  string(REPLACE ${PFX} "" EXE_NAME ${TARGET_NAME})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXE_NAME})
endforeach()

# Testing.
# The checkpointing tests can be run separately using the target
# "test_checkpointing" which builds the examples and runs:
#   ctest -R checkpointing

if (MFEM_ENABLE_TESTING)
  # Short runs, one per checkpoint storage layout, plus one run of the reverse
  # sweep without a window.
  set(EX16_TEST_OPTS -no-vis -tf 0.2 -ci 4 -cw 4)
  set(EX16P_TEST_OPTS ${EX16_TEST_OPTS} -rs 1)

  foreach(STORAGE memory-block file-block memory-snapshots file-snapshots)
    add_test(NAME ${PFX}ex16_${STORAGE}_ser
      COMMAND ${PFX}ex16 ${EX16_TEST_OPTS} -cs ${STORAGE})
  endforeach()
  add_test(NAME ${PFX}ex16_memory-snapshots_no-window_ser
    COMMAND ${PFX}ex16 ${EX16_TEST_OPTS} -cs memory-snapshots -cw 0)

  if (MFEM_USE_MPI)
    foreach(STORAGE memory-block file-snapshots)
      add_test(NAME ${PFX}ex16p_${STORAGE}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${PFX}ex16p> ${EX16P_TEST_OPTS} -cs ${STORAGE}
        ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
endif(MFEM_ENABLE_TESTING)
