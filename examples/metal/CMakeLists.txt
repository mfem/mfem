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

set(METAL_EXAMPLES_SRCS)
list(APPEND METAL_EXAMPLES_SRCS
    axpy.cpp
    )

message(STATUS "[🤘METAL🤘] Adding target: metal")
add_custom_target(metal ${CMAKE_CTEST_COMMAND} -R metal USES_TERMINAL)

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add one executable per cpp file
set(PREFIX metal_)
add_mfem_examples(METAL_EXAMPLES_SRCS ${PREFIX} "" metal)

# Add a test for each example 
if (MFEM_ENABLE_TESTING)
  foreach(SRC_FILE ${METAL_EXAMPLES_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)
    string(REPLACE ".cpp" "" TEST_NAME ${PREFIX}${SRC_FILENAME})

    # add_dependencies(${TEST_NAME} MFEM_METAL_LIB)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    message(STATUS "[🤘METAL🤘] Adding test: ${TEST_NAME}")
  endforeach()
endif()
