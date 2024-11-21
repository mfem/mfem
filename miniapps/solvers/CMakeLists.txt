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

if (MFEM_USE_MPI)
  add_mfem_miniapp(block-solvers
    MAIN block-solvers.cpp
    EXTRA_SOURCES darcy_solver.cpp div_free_solver.cpp bramble_pasciak.cpp
    EXTRA_HEADERS darcy_solver.hpp div_free_solver.hpp bramble_pasciak.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(plor_solvers
    MAIN plor_solvers.cpp
    EXTRA_HEADERS lor_mms.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(lor_elast
    MAIN lor_elast.cpp
    EXTRA_SOURCES block_fespace_operator.cpp
    EXTRA_HEADERS block_fespace_operator.hpp
    LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    if (NOT (HYPRE_USING_CUDA OR HYPRE_USING_HIP))
      # otherwise block-solvers returns MFEM_SKIP_RETURN_VALUE
      add_test(NAME block-solvers-constant_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:block-solvers> -rs 1 -rp 1 -no-vis
        ${MPIEXEC_POSTFLAGS})

      add_test(NAME block-solvers-anisotropic_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:block-solvers> -rs 1 -rp 1
        -m ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.mesh
        -c ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.coeff
        -eb ${CMAKE_CURRENT_SOURCE_DIR}/anisotropic.brd
        -no-vis
        ${MPIEXEC_POSTFLAGS})
    endif()

    add_test(NAME plor_solvers_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:plor_solvers> -fe n -m ../../data/fichera.mesh -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME lor_elast-tri_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lor_elast> -l 2 -ca -pa -ss
      -m ../../data/beam-tri.mesh -o 2
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME lor_elast-hex_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lor_elast> -l 1 -pa -o 2
      -m ../../data/beam-hex.mesh
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()

add_mfem_miniapp(lor_solvers
  MAIN lor_solvers.cpp
  EXTRA_HEADERS lor_mms.hpp
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME lor_solvers_ser COMMAND lor_solvers -fe n -no-vis)
endif()
