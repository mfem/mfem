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
  add_mfem_miniapp(generate_random_field
    MAIN generate_random_field.cpp
    EXTRA_SOURCES material_metrics.cpp spde_solver.cpp util.cpp visualizer.cpp
                  transformation.cpp
    LIBRARIES mfem)

  # Copy files to the build folder.
  file(COPY ${PROJECT_SOURCE_DIR}/miniapps/spde/README.md
       DESTINATION ${PROJECT_BINARY_DIR}/miniapps/spde)

  if (MFEM_ENABLE_TESTING)
    if (MFEM_USE_DOUBLE)
      # otherwise returns MFEM_SKIP_RETURN_VALUE
      add_test(NAME generate_random_field_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:generate_random_field> -no-vis
        -r 3 -rp 2 -nu 4 -l1 0.09 -l2 0.03 -l3 0.05 -s 0.01 -t 0.08 -top 1
        -no-rs -m ../../data/ref-square.mesh
        ${MPIEXEC_POSTFLAGS})
    endif()
  endif()
endif()
