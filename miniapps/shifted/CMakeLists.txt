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

  add_mfem_miniapp(lsf_integral
    MAIN lsf_integral.cpp
    ${DIST_COMMON_FILES}
    LIBRARIES mfem-common)

  if (MFEM_ENABLE_TESTING)
    set(PARALLEL_TESTS
      distance
      diffusion
      extrapolate
      lsf_integral
    )
    # Shifted miniapps that return MFEM_SKIP_RETURN_VALUE in some cases:
    set(SKIP_TESTS)
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
