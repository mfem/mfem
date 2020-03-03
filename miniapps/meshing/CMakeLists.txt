# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(klein-bottle
  MAIN klein-bottle.cpp
  LIBRARIES mfem)

add_mfem_miniapp(mesh-explorer
  MAIN mesh-explorer.cpp
  LIBRARIES mfem)

add_mfem_miniapp(mobius-strip
  MAIN mobius-strip.cpp
  LIBRARIES mfem)

add_mfem_miniapp(shaper
  MAIN shaper.cpp
  LIBRARIES mfem)

add_mfem_miniapp(extruder
  MAIN extruder.cpp
  LIBRARIES mfem)

add_mfem_miniapp(mesh-optimizer
  MAIN mesh-optimizer.cpp
  LIBRARIES mfem)

add_mfem_miniapp(toroid
  MAIN toroid.cpp
  LIBRARIES mfem)

# Add serial tests.
add_test(NAME mesh-optimizer
  COMMAND mesh-optimizer -no-vis -m ${CMAKE_CURRENT_SOURCE_DIR}/icf.mesh)

# Parallel apps.
if (MFEM_USE_MPI)
  add_mfem_miniapp(pmesh-optimizer
    MAIN pmesh-optimizer.cpp
    LIBRARIES mfem)

  # Add parallel tests.
  add_test(NAME pmesh-optimizer_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:pmesh-optimizer> -no-vis
    -m ${CMAKE_CURRENT_SOURCE_DIR}/icf.mesh
    ${MPIEXEC_POSTFLAGS})
endif()
