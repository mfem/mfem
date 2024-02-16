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
  list(APPEND HOOKE_SOURCES
      operators/elasticity_operator.cpp
      operators/elasticity_gradient_operator.cpp
      preconditioners/diagonal_preconditioner.cpp)
  list(APPEND HOOKE_HEADERS
      operators/elasticity_operator.hpp
      operators/elasticity_gradient_operator.hpp
      preconditioners/diagonal_preconditioner.hpp
      materials/gradient_type.hpp
      materials/linear_elastic.hpp
      materials/neohookean.hpp
      kernels/elasticity_kernels.hpp
      kernels/kernel_helpers.hpp)

  convert_filenames_to_full_paths(HOOKE_SOURCES)
  convert_filenames_to_full_paths(HOOKE_HEADERS)

  set(HOOKE_COMMON_FILES
      EXTRA_SOURCES ${HOOKE_SOURCES}
      EXTRA_HEADERS ${HOOKE_HEADERS})

  # The hooke miniapp crashes the AMD HIP compiler, so for now we disable it
  # when HIP is enabled:
  if (NOT MFEM_USE_HIP)
    add_mfem_miniapp(hooke
      MAIN hooke.cpp
      ${HOOKE_COMMON_FILES}
      LIBRARIES mfem)

    # Add parallel tests.
    if (MFEM_ENABLE_TESTING)
      add_test(NAME hooke_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:hooke> -rs 0 -no-vis
        ${MPIEXEC_POSTFLAGS})
    endif()
  endif()
endif ()
