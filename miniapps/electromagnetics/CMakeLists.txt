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

if (MFEM_USE_MPI)
  add_mfem_miniapp(tesla
    MAIN tesla.cpp
    EXTRA_SOURCES tesla_solver.cpp
    EXTRA_HEADERS tesla_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  add_mfem_miniapp(volta
    MAIN volta.cpp
    EXTRA_SOURCES volta_solver.cpp
    EXTRA_HEADERS volta_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

 add_mfem_miniapp(joule
   MAIN joule.cpp
   EXTRA_SOURCES joule_solver.cpp
   EXTRA_HEADERS joule_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
   LIBRARIES mfem mfem_miniapps_common)

 add_mfem_miniapp(maxwell
   MAIN maxwell.cpp
   EXTRA_SOURCES maxwell_solver.cpp
   EXTRA_HEADERS maxwell_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
   LIBRARIES mfem mfem_miniapps_common)

  # Add the corresponding tests to the "test" target
  add_test(NAME tesla_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:tesla> -no-vis -maxit 2 -cr "0 0 -0.2 0 0 0.2 0.2 0.4 1"
    ${MPIEXEC_POSTFLAGS})

  add_test(NAME volta_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:volta> -no-vis -maxit 2 -dbcs 1 -dbcg -ds "0.0 0.0 0.0 0.2 8.0"
    ${MPIEXEC_POSTFLAGS})

 add_test(NAME joule_np=4
   COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
   ${MPIEXEC_PREFLAGS}
   $<TARGET_FILE:joule>
   -no-vis -p rod -tf 3 -m ${CMAKE_CURRENT_SOURCE_DIR}/cylinder-hex.mesh
   ${MPIEXEC_POSTFLAGS})

 add_test(NAME maxwell_np=4
   COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
   ${MPIEXEC_PREFLAGS}
   $<TARGET_FILE:maxwell>
   -no-vis -abcs "-1" -dp "-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5"
   ${MPIEXEC_POSTFLAGS})

endif()
