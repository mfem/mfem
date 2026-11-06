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

list(APPEND OPTIMIZATION_COMMON_SOURCES
  optproblem.cpp
  functionals.cpp)
list(APPEND OPTIMIZATION_COMMON_HEADERS
  optproblem.hpp
  functionals.hpp)

convert_filenames_to_full_paths(OPTIMIZATION_COMMON_SOURCES)
convert_filenames_to_full_paths(OPTIMIZATION_COMMON_HEADERS)

set(OPTIMIZATION_COMMON_FILES
  EXTRA_SOURCES ${OPTIMIZATION_COMMON_SOURCES}
  EXTRA_HEADERS ${OPTIMIZATION_COMMON_HEADERS})

add_mfem_miniapp(optproblem_test
  MAIN optproblem_test.cpp
  ${OPTIMIZATION_COMMON_FILES}
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME optimization_optproblem_test COMMAND optproblem_test)
  if (MFEM_USE_MPI)
    add_test(NAME optimization_optproblem_test_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:optproblem_test>
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
