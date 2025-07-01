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

if(MFEM_USE_MPI)
  add_custom_target(clean_common
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning common directory..."
    COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/../common clean
    COMMENT "Cleaning common directory..."
  )

  # Define miniapps
  set(MINIAPPS
    abs-l1-jacobi
    mg-abs-l1-jacobi
  )

  # Add miniapps
  foreach(APP ${MINIAPPS})
    add_mfem_miniapp(${APP}
      MAIN ${APP}.cpp
      ${MFEM_MINIAPPS_COMMON_HEADERS}
      EXTRA_SOURCES ds-common.cpp
      EXTRA_HEADERS ds-common.hpp
      LIBRARIES mfem-common mfem
    )
  endforeach()

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME abs-l1-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:abs-l1-jacobi> -m ../../data/ref-cube.mesh -rs 2 -rp 2
      -s 1 -i 1 -a 3 -pc 1 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME mg-abs-l1-jacobi_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mg-abs-l1-jacobi> -m ../../data/ref-cube.mesh -rs 2 -rp 1
      -ol 1 -gl 1 -s 1 -i 1 -a 3 -no-mon -no-vis
      ${MPIEXEC_POSTFLAGS})
  endif()

  # Add custom target for cleaning everything
  add_custom_target(clean_all
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_CURRENT_SOURCE_DIR}/../common clean
    COMMENT "Cleaning all build artifacts..."
  )
endif()
