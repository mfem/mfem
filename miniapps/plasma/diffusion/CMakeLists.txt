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

if (MFEM_USE_MPI)
  add_mfem_miniapp(umansky_cg
    MAIN umansky_cg.cpp
    EXTRA_HEADERS umansky.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)

  add_mfem_miniapp(umansky_dg
    MAIN umansky_cg.cpp
    EXTRA_HEADERS umansky.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)

# Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME umansky_cg_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:umansky_cg> -no-vis -maxit 2 -rs 1 -rp 1 --height 0.8 --width 1.2 -Ak 1e2
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME umansky_dg_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:umansky_dg> -no-vis -maxit 2 -rs 1 -rp 1 --height 0.8 --width 1.2 -Ak 1e2 -k 5
      ${MPIEXEC_POSTFLAGS})

  endif()
endif()
