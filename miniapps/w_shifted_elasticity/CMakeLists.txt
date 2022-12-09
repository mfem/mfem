# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
  list(APPEND LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_SOURCES
        nitsche_weighted_solver.cpp
	shifted_weighted_solver.cpp
	volume_fractions.cpp
	volume_weighted_integrals.cpp
	AnalyticalGeometricShape.cpp
	ghost_penalty.cpp
	marking.cpp
	dist_solver.cpp
	linear_elasticity_solver.cpp)
  list(APPEND LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_HEADERS
        nitsche_weighted_solver.hpp
	shifted_weighted_solver.hpp
	volume_fractions.hpp
	volume_weighted_integrals.hpp
	AnalyticalGeometricShape.hpp
	ghost_penalty.hpp
	marking.hpp
	dist_solver.hpp
	linear_elasticity_solver.hpp
	sbm_aux.hpp)

convert_filenames_to_full_paths(LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_SOURCES)
convert_filenames_to_full_paths(LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_HEADERS)

set(LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_FILES
      EXTRA_SOURCES ${LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_SOURCES}
      EXTRA_HEADERS ${LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_HEADERS})

  add_mfem_miniapp(ex_Linear_Elasticity_WSBM
    MAIN ex_Linear_Elasticity_refactored.cpp
    ${LINEAR_ELASTICITY_WEIGHTED_SHIFTED_COMMON_FILES}
    LIBRARIES mfem mfem-common)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME ex_Linear_Elasticity_WSBM_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ex_Linear_Elasticity_WSBM> -no-vis
      ${MPIEXEC_POSTFLAGS})
      
  endif()
endif ()

