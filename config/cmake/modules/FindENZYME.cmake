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

message(STATUS "Looking for ENZYME ...")
message(STATUS "   in ENZYME_DIR = ${ENZYME_DIR}")

# Make sure the directory and version combination works. Do nothing otherwise.
if(EXISTS "${ENZYME_DIR}/ClangEnzyme-${ENZYME_VERSION}.so")
  message(STATUS "Found ENZYME: ${ENZYME_DIR}/ClangEnzyme-${ENZYME_VERSION}.so")

  # Set ENZYME_FOUND
  set(ENZYME_FOUND TRUE CACHE BOOL "ENZYME was found." FORCE)

  # Set CXX flags to accommodate the Enzyme Clang plugin
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -load -Xclang ${ENZYME_DIR}/ClangEnzyme-${ENZYME_VERSION}.so -mllvm -enzyme-loose-types=1")
  set(MFEM_USE_ENZYME YES)
else()

endif()
