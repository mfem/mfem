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

add_mfem_miniapp(stab_condif
  MAIN stab_condif.cpp
  LIBRARIES mfem)


if (MFEM_ENABLE_TESTING)
  add_test(NAME stab_condif
    COMMAND $<TARGET_FILE:stab_condif> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2)
endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(stab_navsto
    MAIN stab_navsto.cpp
    LIBRARIES mfem)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME stab_navsto_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:stab_navsto> -no-vis
      ${MPIEXEC_POSTFLAGS}
      -m ${PROJECT_SOURCE_DIR}/data/square-nurbs.mesh -r 1 -o 2)
  endif()
endif()
