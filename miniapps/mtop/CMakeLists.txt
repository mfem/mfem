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

# Skip dFEM qfunction with floats
if(MFEM_USE_MPI AND MFEM_USE_DOUBLE)

	list(APPEND PAR_MTOP_SOURCES mtop_solvers.cpp)
	list(APPEND PAR_MTOP_SOURCES linear_elasticity_solver.cpp)
	list(APPEND PAR_MTOP_HEADERS mtop_solvers.hpp)
	list(APPEND PAR_MTOP_HEADERS linear_elasticity_solver.hpp)

	convert_filenames_to_full_paths(PAR_MTOP_SOURCES)
	convert_filenames_to_full_paths(PAR_MTOP_HEADERS)

	set(PAR_MTOP_COMMON_FILES
		EXTRA_SOURCES ${PAR_MTOP_SOURCES}
		EXTRA_HEADERS ${PAR_MTOP_HEADERS})

	add_mfem_miniapp(mtop_test_iso_elasticity
					 MAIN mtop_test_iso_elasticity.cpp
					 ${PAR_MTOP_COMMON_FILES}
					 LIBRARIES mfem)

	add_mfem_miniapp(mtop_ex_linear_elasticity_solver
					 MAIN mtop_ex_linear_elasticity_solver.cpp
					 ${PAR_MTOP_COMMON_FILES}
					 LIBRARIES mfem)

	if (MFEM_ENABLE_TESTING)
		add_test(NAME mtop_linear_elasticity_solver_2d_np${MFEM_MPI_NP}
			COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
			${MPIEXEC_PREFLAGS}
			$<TARGET_FILE:mtop_ex_linear_elasticity_solver>
			-m ${CMAKE_CURRENT_SOURCE_DIR}/sq_2D_9_quad.mesh -rp 0 -no-pv
			${MPIEXEC_POSTFLAGS})
		add_test(NAME mtop_linear_elasticity_solver_3d_np${MFEM_MPI_NP}
			COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
			${MPIEXEC_PREFLAGS}
			$<TARGET_FILE:mtop_ex_linear_elasticity_solver>
			-m ${PROJECT_SOURCE_DIR}/data/inline-hex.mesh -rp 0 -no-pv
			${MPIEXEC_POSTFLAGS})
	endif()

endif (MFEM_USE_MPI AND MFEM_USE_DOUBLE)
