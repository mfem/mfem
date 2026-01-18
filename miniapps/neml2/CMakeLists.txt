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

find_package(neml2 CONFIG)

if(MFEM_USE_MPI AND neml2_FOUND)
  list(APPEND MFEM_NEML2_SOURCES
    dfem-neml2.cpp
    operators.cpp
    constitutive.cpp)
  list(APPEND MFEM_NEML2_HEADERS
    operators.hpp
    constitutive.hpp)

  convert_filenames_to_full_paths(MFEM_NEML2_SOURCES)
  convert_filenames_to_full_paths(MFEM_NEML2_HEADERS)

  set(NEML2_COMMON_FILES
    EXTRA_SOURCES ${MFEM_NEML2_SOURCES}
    EXTRA_HEADERS ${MFEM_NEML2_HEADERS})

  add_mfem_miniapp(dfem-neml2
    MAIN dfem-neml2.cpp
    ${NEML2_COMMON_FILES}
    LIBRARIES mfem)

  # Link neml2
  target_link_libraries(dfem-neml2 PRIVATE neml2::neml2)

  # Add parallel tests.
  if(MFEM_ENABLE_TESTING)
    add_test(NAME neml2_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dfem-neml2>
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
