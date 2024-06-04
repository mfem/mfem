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


list(APPEND STAB_COMMON_SOURCES
     stab_condif.cpp stab_navsto.cpp)
list(APPEND STAB_COMMON_HEADERS
     stab_condif.hpp stab_navsto.hpp manu.hpp skew.hpp)

set(STAB_COMMON_FILES
    EXTRA_SOURCES ${STAB_COMMON_SOURCES}
    EXTRA_HEADERS ${STAB_COMMON_HEADERS})

add_mfem_miniapp(condif
    MAIN condif.cpp
    ${STAB_COMMON_FILES}
    LIBRARIES mfem)

add_mfem_miniapp(navsto
    MAIN navsto.cpp
    ${STAB_COMMON_FILES}
    LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME ex_condif
    COMMAND $<TARGET_FILE:condif> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2)
endif()

if (MFEM_USE_MPI)
  #add_mfem_miniapp(stab_navsto_par
  #  MAIN stab_navsto_par.cpp
  #  LIBRARIES mfem)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME navsto_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:navsto> -no-vis
      ${MPIEXEC_POSTFLAGS}
      -m ${PROJECT_SOURCE_DIR}/data/square-nurbs.mesh -r 1 -o 2)
  endif()
endif()
