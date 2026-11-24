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

# opt_test.cpp uses the previous optimization API and is not built for now.

list(APPEND OPTIMIZATION_SOURCES
  optproblem.cpp)
list(APPEND OPTIMIZATION_HEADERS
  functional.hpp
  optproblem.hpp)

convert_filenames_to_full_paths(OPTIMIZATION_SOURCES)
convert_filenames_to_full_paths(OPTIMIZATION_HEADERS)

add_mfem_miniapp(optproblem_test
  MAIN optproblem_test.cpp
  EXTRA_SOURCES ${OPTIMIZATION_SOURCES}
  EXTRA_HEADERS ${OPTIMIZATION_HEADERS}
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME optproblem_test COMMAND optproblem_test)
endif()
