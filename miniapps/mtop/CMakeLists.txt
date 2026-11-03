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
	list(APPEND PAR_MTOP_HEADERS mtop_solvers.hpp)

	convert_filenames_to_full_paths(PAR_MTOP_SOURCES)
	convert_filenames_to_full_paths(PAR_MTOP_HEADERS)

	set(PAR_MTOP_COMMON_FILES
		EXTRA_SOURCES ${PAR_MTOP_SOURCES}
		EXTRA_HEADERS ${PAR_MTOP_HEADERS})

	add_mfem_miniapp(mtop_test_iso_elasticity
					 MAIN mtop_test_iso_elasticity.cpp
					 ${PAR_MTOP_COMMON_FILES}
					 LIBRARIES mfem)

endif (MFEM_USE_MPI AND MFEM_USE_DOUBLE)

if (MFEM_USE_LAPACK)
  add_mfem_miniapp(projection_ghost_penalty
    MAIN projection_ghost_penalty.cpp
    EXTRA_SOURCES
    EXTRA_HEADERS
    LIBRARIES mfem)
  if (MFEM_ENABLE_TESTING)
    add_test(NAME ghost_penalty_demo_serial
      COMMAND projection_ghost_penalty --serial --order 2 --vdim 3)
    set_tests_properties(ghost_penalty_demo_serial PROPERTIES
      LABELS ghost-penalty TIMEOUT 120)
    foreach(dimension 2 3)
      foreach(space auto p q)
        add_test(NAME ghost_penalty_demo_tensor_${dimension}d_${space}_serial
          COMMAND projection_ghost_penalty --serial --element-type tensor
          --dimension ${dimension} --projection-space ${space} --order 2 --vdim 3)
        set_tests_properties(ghost_penalty_demo_tensor_${dimension}d_${space}_serial
          PROPERTIES LABELS ghost-penalty TIMEOUT 120)
      endforeach()
    endforeach()
    if (MFEM_USE_MPI)
      foreach(ranks 1 2 4)
        add_test(NAME ghost_penalty_demo_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:projection_ghost_penalty>
          --parallel --order 2 --vdim 3 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(ghost_penalty_demo_np${ranks} PROPERTIES
          LABELS ghost-penalty TIMEOUT 120 PROCESSORS ${ranks})
        foreach(dimension 2 3)
          add_test(NAME ghost_penalty_demo_tensor_${dimension}d_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:projection_ghost_penalty>
            --parallel --element-type tensor --dimension ${dimension}
            --order 2 --vdim 3 ${MPIEXEC_POSTFLAGS})
          set_tests_properties(ghost_penalty_demo_tensor_${dimension}d_np${ranks}
            PROPERTIES LABELS ghost-penalty TIMEOUT 120 PROCESSORS ${ranks})
        endforeach()
      endforeach()
    endif()
  endif()
endif()
