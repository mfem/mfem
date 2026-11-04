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
  add_mfem_miniapp(projection_ghost_penalty_serial
    MAIN projection_ghost_penalty_serial.cpp
    EXTRA_SOURCES
    EXTRA_HEADERS
    LIBRARIES mfem)
  if (MFEM_USE_MPI)
    add_mfem_miniapp(projection_ghost_penalty_parallel
      MAIN projection_ghost_penalty_parallel.cpp
      EXTRA_SOURCES
      EXTRA_HEADERS
      LIBRARIES mfem)
  endif()
  if (MFEM_ENABLE_TESTING)
    add_test(NAME ghost_penalty_demo_serial
      COMMAND projection_ghost_penalty_serial --order 2 --vdim 3)
    set_tests_properties(ghost_penalty_demo_serial PROPERTIES
      LABELS ghost-penalty TIMEOUT 120)
    foreach(dimension 2 3)
      foreach(space auto p q)
        add_test(NAME ghost_penalty_demo_tensor_${dimension}d_${space}_serial
          COMMAND projection_ghost_penalty_serial --element-type tensor
          --dimension ${dimension} --projection-space ${space} --order 2 --vdim 3)
        set_tests_properties(ghost_penalty_demo_tensor_${dimension}d_${space}_serial
          PROPERTIES LABELS ghost-penalty TIMEOUT 120)
      endforeach()
    endforeach()
    foreach(dimension 2 3)
      foreach(family simplex tensor)
        foreach(geometry_order 2 3)
          set(case ghost_penalty_demo_curved_${family}_${dimension}d_g${geometry_order})
          add_test(NAME ${case}
            COMMAND projection_ghost_penalty_serial --element-type ${family}
            --dimension ${dimension} --geometry-order ${geometry_order}
            --warp 0.05 --projection-space p --projection-degree 1 --order 2)
          set_tests_properties(${case} PROPERTIES
            LABELS "ghost-penalty;ghost-curved" TIMEOUT 120)
        endforeach()
      endforeach()
      add_test(NAME ghost_penalty_demo_warped_${dimension}d
        COMMAND projection_ghost_penalty_serial --element-type tensor
        --dimension ${dimension} --geometry-order 1 --warp 0.05
        --projection-space p --order 2)
      set_tests_properties(ghost_penalty_demo_warped_${dimension}d PROPERTIES
        LABELS "ghost-penalty;ghost-curved" TIMEOUT 120)
    endforeach()
    if (MFEM_USE_MPI)
      foreach(ranks 1 2 4)
        add_test(NAME ghost_penalty_demo_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:projection_ghost_penalty_parallel>
          --order 2 --vdim 3 ${MPIEXEC_POSTFLAGS})
        set_tests_properties(ghost_penalty_demo_np${ranks} PROPERTIES
          LABELS ghost-penalty TIMEOUT 120 PROCESSORS ${ranks})
        foreach(dimension 2 3)
          foreach(family simplex tensor)
            set(case ghost_penalty_demo_curved_${family}_${dimension}d_np${ranks})
            add_test(NAME ${case}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:projection_ghost_penalty_parallel>
              --element-type ${family} --dimension ${dimension}
              --geometry-order 2 --warp 0.05 --projection-space p
              --projection-degree 1 --order 2 --vdim 3 ${MPIEXEC_POSTFLAGS})
            set_tests_properties(${case} PROPERTIES
              LABELS "ghost-penalty;ghost-curved" TIMEOUT 120 PROCESSORS ${ranks})
          endforeach()
          add_test(NAME ghost_penalty_demo_tensor_${dimension}d_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:projection_ghost_penalty_parallel>
            --element-type tensor --dimension ${dimension}
            --order 2 --vdim 3 ${MPIEXEC_POSTFLAGS})
          set_tests_properties(ghost_penalty_demo_tensor_${dimension}d_np${ranks}
            PROPERTIES LABELS ghost-penalty TIMEOUT 120 PROCESSORS ${ranks})
        endforeach()
      endforeach()
      find_package(Python3 3.7 QUIET COMPONENTS Interpreter)
      if (Python3_Interpreter_FOUND)
        set(ghost_precision single)
        if (MFEM_USE_DOUBLE)
          set(ghost_precision double)
        endif()
        set(ghost_mpi_postflags)
        foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
          list(APPEND ghost_mpi_postflags "--mpi-postflag=${flag}")
        endforeach()
        foreach(ranks 1 2 4)
          add_test(NAME ghost_penalty_demo_compare_np${ranks}
            COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/unit/fem/ghost_penalty_compare.py
            --serial $<TARGET_FILE:projection_ghost_penalty_serial>
            --parallel $<TARGET_FILE:projection_ghost_penalty_parallel>
            --ranks ${ranks} --precision ${ghost_precision}
            ${ghost_mpi_postflags} --launcher
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS})
          set_tests_properties(ghost_penalty_demo_compare_np${ranks} PROPERTIES
            LABELS ghost-penalty TIMEOUT 180 PROCESSORS ${ranks})
        endforeach()
      else()
        message(STATUS "Python3 unavailable; skipping ghost demo comparisons")
      endif()
    endif()
  endif()
endif()
