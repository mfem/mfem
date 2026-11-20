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

if (MFEM_USE_MPI)
  add_mfem_miniapp(tesla
    MAIN tesla.cpp
    tesla_solver.cpp
    EXTRA_HEADERS tesla_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)

  add_mfem_miniapp(volta
    MAIN volta.cpp
    volta_solver.cpp
    EXTRA_HEADERS volta_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)

 add_mfem_miniapp(joule
   MAIN joule.cpp
   joule_solver.cpp
   EXTRA_HEADERS joule_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
   LIBRARIES mfem-common)

 add_mfem_miniapp(maxwell
   MAIN maxwell.cpp
   maxwell_solver.cpp
   EXTRA_HEADERS maxwell_solver.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
   LIBRARIES mfem-common)

 if (MFEM_USE_GSLIB)
  add_mfem_miniapp(lorentz
    MAIN lorentz.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)
endif()
  
  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME tesla_1_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:tesla> -no-vis -maxit 2 -cr "0 0 -0.2 0 0 0.2 0.2 0.4 1"
      ${MPIEXEC_POSTFLAGS})
    add_test(NAME tesla_2_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:tesla>
      -no-vis -maxit 2 -m ../../data/inline-hex.mesh -ubbc "0 0 1"
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME volta_1_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:volta>
      -no-vis -maxit 2 -dbcs 1 -dbcg -ds "0.0 0.0 0.0 0.2 8.0"
      ${MPIEXEC_POSTFLAGS})
    add_test(NAME volta_2_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:volta>
      -no-vis -maxit 2 -m ../../data/square-disc.mesh -dbcs "1 2 3 4 5 6 7 8"
      -dbcv "0 0 0 0 1 1 1 1"
      ${MPIEXEC_POSTFLAGS})
    add_test(NAME volta_3_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:volta>
      -no-vis -maxit 2 -m ../../data/inline-hex.mesh -dbcs "1 6" -dbcv "0 1"
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME joule_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:joule>
      -no-vis -p rod -tf 3 -m ${CMAKE_CURRENT_SOURCE_DIR}/cylinder-hex.mesh
      ${MPIEXEC_POSTFLAGS})

    if (MFEM_USE_DOUBLE) # otherwise returns MFEM_SKIP_RETURN_VALUE
      add_test(NAME maxwell_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:maxwell>
        -no-vis -abcs "-1" -dp "-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5"
        ${MPIEXEC_POSTFLAGS})
    endif()

    if (MFEM_USE_GSLIB)
      add_test(NAME lorentz_1_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:lorentz>
        -no-vis -er Volta-AMR-Parallel -ec 2 -npt 100 -xmin "0.0 0.0 0.0"
        -xmax "1.0 1.0 1.0" -pmin "1 0 0" -pmax "1 0 0" -rdf 0 -vt 0 -nt 100
        ${MPIEXEC_POSTFLAGS})
      # Setup dependency on volta_3_np=<np>
      set_tests_properties(volta_3_np=${MFEM_MPI_NP}
        PROPERTIES FIXTURES_SETUP Volta3)
      set_tests_properties(lorentz_1_np=${MFEM_MPI_NP}
        PROPERTIES FIXTURES_REQUIRED Volta3)

      add_test(NAME lorentz_2_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:lorentz>
        -no-vis -br Tesla-AMR-Parallel -bc 2 -npt 10 -xmin "0.0 0.0 0.0"
        -xmax "1.0 1.0 1.0" -pmin "0 0.1 0.05" -pmax "0 0.4 0.1" -nt 1000 -rdf 0
        -vt 0
        ${MPIEXEC_POSTFLAGS})
      # Setup dependency on tesla_2_np=<np>
      set_tests_properties(tesla_2_np=${MFEM_MPI_NP}
        PROPERTIES FIXTURES_SETUP Tesla2)
      set_tests_properties(lorentz_2_np=${MFEM_MPI_NP}
        PROPERTIES FIXTURES_REQUIRED Tesla2)
    endif()
  endif()
endif()
