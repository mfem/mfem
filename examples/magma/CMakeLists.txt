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

set(MAGMA_EXAMPLES_SRCS)
list(APPEND MAGMA_EXAMPLES_SRCS
  batched_mass_bench.cpp
  )

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_magma" target (build-only by default).
add_custom_target(test_magma
  COMMENT "Building MAGMA examples ...")

# Add one executable per cpp file, adding "magma_" as prefix to avoid
# collisions in a unified build tree.
set(PFX magma_)
add_mfem_examples(MAGMA_EXAMPLES_SRCS ${PFX} "" test_magma)
