# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

set(CALIPER_EXAMPLES_SRCS)

list(APPEND CALIPER_EXE_SRCS
    ex1.cpp
    )

if (MFEM_USE_MPI)
  list(APPEND CALIPER_EXE_SRCS
      ex1p.cpp
      )
endif()


# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add one executable per cpp file
set(PREFIX caliper_)
add_mfem_examples(CALIPER_EXE_SRCS ${PREFIX})

# Add a test for each example
if (MFEM_ENABLE_TESTING)
  foreach(SRC_FILE ${CALIPER_EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    string(REPLACE ".cpp" "" TEST_NAME ${PREFIX}${SRC_FILENAME})

    set(THIS_TEST_OPTIONS "-no-vis")

    if (NOT (${TEST_NAME} MATCHES ".*p$"))
      add_test(NAME ${TEST_NAME}_ser
        COMMAND $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS})
    else()
      add_test(NAME ${TEST_NAME}_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${TEST_NAME}> ${THIS_TEST_OPTIONS}
        ${MPIEXEC_POSTFLAGS})
    endif()
  endforeach()
endif()
