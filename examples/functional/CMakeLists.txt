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

set(FUNCTIONAL_EXAMPLES_SRCS)
list(APPEND FUNCTIONAL_EXAMPLES_SRCS
  ex0.cpp
  ex1.cpp
  ex2.cpp
)
if (MFEM_USE_MPI)
  list(APPEND FUNCTIONAL_EXAMPLES_SRCS
    ex0p.cpp
    ex1p.cpp
  )
)

# Include the source directory where mfem.hpp and mfem-performance.hpp are.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# Add "test_ginkgo" target, see below.
add_custom_target(test_ginkgo
  ${CMAKE_CTEST_COMMAND} -R ginkgo USES_TERMINAL)

# Add one executable per cpp file, adding "ginkgo_" as prefix. Sets
# "test_ginkgo" as a target that depends on the given examples.
set(PREFIX functional_)
add_mfem_examples(FUNCTIONAL_EXAMPLES_SRCS ${PREFIX})
