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

# Serial miniapps
add_mfem_miniapp(lsf_integral
  MAIN lsf_integral.cpp
  LIBRARIES mfem-common)

# Serial tests
if (MFEM_ENABLE_TESTING AND MFEM_USE_ALGOIM)
  set(SERIAL_TESTS
    lsf_integral
  )
  foreach(test ${SERIAL_TESTS})
    add_test(NAME ${test} COMMAND ${test} -no-vis)
  endforeach()
endif()

# Parallel miniapps and tests
if (MFEM_USE_MPI)
  add_mfem_miniapp(lsf_integralp
    MAIN lsf_integral.cpp
    LIBRARIES mfem-common)
  target_compile_definitions(lsf_integralp PRIVATE MFEM_CUT_INTEGRAL_PARALLEL)
  if (MFEM_ENABLE_TESTING AND MFEM_USE_ALGOIM)
    foreach(dimension 2 3)
      foreach(ranks 1 2 4)
        add_test(NAME cut_plane_${dimension}d_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:lsf_integralp>
          -dim ${dimension} -ls 2 -no-vis ${MPIEXEC_POSTFLAGS})
        set_tests_properties(cut_plane_${dimension}d_np${ranks}
          PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
        add_test(NAME cut_general_${dimension}d_np${ranks}
          COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:lsf_integralp>
          -dim ${dimension} -ls 2 -cq algoim-general -no-vis ${MPIEXEC_POSTFLAGS})
        set_tests_properties(cut_general_${dimension}d_np${ranks}
          PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
        if (MFEM_USE_LAPACKE)
          add_test(NAME cut_two_planes_${dimension}d_np${ranks}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:lsf_integralp>
            -dim ${dimension} -ls 3 -no-vis ${MPIEXEC_POSTFLAGS})
          set_tests_properties(cut_two_planes_${dimension}d_np${ranks}
            PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
        endif()
      endforeach()
    endforeach()
    add_test(NAME cut_plane_empty_partitions
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lsf_integralp> -dim 2 -n 1 -ls 2 -no-vis
      ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cut_plane_empty_partitions
      PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
    add_test(NAME cut_general_empty_partitions
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lsf_integralp> -dim 2 -n 1 -ls 2 -cq algoim-general -no-vis
      ${MPIEXEC_POSTFLAGS})
    add_test(NAME cut_general_collective_failure
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:lsf_integralp> -dim 2 -n 1 -ls 3 -cq algoim-general -no-vis
      ${MPIEXEC_POSTFLAGS})
    set_tests_properties(cut_general_empty_partitions cut_general_collective_failure
      PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
    set_tests_properties(cut_general_collective_failure PROPERTIES WILL_FAIL TRUE)
    if (MFEM_USE_LAPACKE)
      add_test(NAME cut_two_planes_empty_partitions
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:lsf_integralp> -dim 2 -n 1 -ls 3 -no-vis
        ${MPIEXEC_POSTFLAGS})
      set_tests_properties(cut_two_planes_empty_partitions
        PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1" TIMEOUT 60)
    endif()
  endif()
  list(APPEND DIST_COMMON_SOURCES
        sbm_solver.cpp
        marking.cpp
        extrapolator.cpp)
  list(APPEND DIST_COMMON_HEADERS
        sbm_solver.hpp
        sbm_aux.hpp
        marking.hpp
        extrapolator.hpp)

  convert_filenames_to_full_paths(DIST_COMMON_SOURCES)
  convert_filenames_to_full_paths(DIST_COMMON_HEADERS)

  set(DIST_COMMON_FILES
      EXTRA_SOURCES ${DIST_COMMON_SOURCES}
      EXTRA_HEADERS ${DIST_COMMON_HEADERS})

  add_mfem_miniapp(distance
    MAIN distance.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem-common)

  add_mfem_miniapp(diffusion
    MAIN diffusion.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem-common)

  add_mfem_miniapp(extrapolate
    MAIN extrapolate.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem-common)

  if (MFEM_ENABLE_TESTING)
    set(PARALLEL_TESTS
      lsf_integralp
      distance
      diffusion
      extrapolate
    )
    # Shifted miniapps that return MFEM_SKIP_RETURN_VALUE in some cases:
    set(SKIP_TESTS)
    if (NOT MFEM_USE_ALGOIM)
      list(APPEND SKIP_TESTS lsf_integralp)
    endif()
    if (HYPRE_USING_CUDA OR HYPRE_USING_HIP)
      list(APPEND SKIP_TESTS diffusion)
    endif()

    foreach(test ${PARALLEL_TESTS})
      if (test IN_LIST SKIP_TESTS)
        continue()
      endif()
      add_test(NAME shifted_${test}_np${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${test}> -no-vis
        ${MPIEXEC_POSTFLAGS})
    endforeach()
  endif()
endif ()
