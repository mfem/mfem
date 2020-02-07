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

set(PERFORMANCE_CXX_OPTIONS)
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  list(APPEND PERFORMANCE_CXX_OPTIONS
    "-march=native"
    "-std=c++11"
    "-Wpedantic"
    "-Wall"
    "-fcolor-diagnostics"
    "-fvectorize"
    "-fslp-vectorize"
    "-fslp-vectorize-aggressive"
    "-ffp-contract=fast")
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
  list(APPEND PERFORMANCE_CXX_OPTIONS
    "-march=native"
    "-std=c++11"
    "-pedantic"
    "-Wall"
    "--param" "max-completely-peel-times=3")
endif()

add_mfem_miniapp(performance_ex1
  MAIN ex1.cpp
  LIBRARIES mfem
  EXTRA_OPTIONS ${PERFORMANCE_CXX_OPTIONS})

add_test(NAME performance_ex1_ser
  COMMAND performance_ex1 -no-vis -r 2)

if (MFEM_USE_MPI)
  add_mfem_miniapp(performance_ex1p
    MAIN ex1p.cpp
    LIBRARIES mfem
    EXTRA_OPTIONS ${PERFORMANCE_CXX_OPTIONS})

  add_test(NAME performance_ex1p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:performance_ex1p> -no-vis -rs 2
    ${MPIEXEC_POSTFLAGS})
endif()
