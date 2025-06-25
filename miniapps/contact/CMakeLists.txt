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

if (MFEM_USE_TRIBOL AND MFEM_USE_MPI)

  list(APPEND CONTACT_SOURCES
      problems/parproblems.cpp
      problems/parproblems_util.cpp
      ipsolver/ParIPsolver.cpp)
  list(APPEND CONTACT_HEADERS
      problems/parproblems.hpp
      problems/parproblems_util.hpp
      ipsolver/ParIPsolver.hpp)

  if (MFEM_USE_MUMPS OR MFEM_USE_MKL_CPARDISO)
    list(APPEND CONTACT_SOURCES
        ipsolver/two-level-solver.cpp)
    list(APPEND CONTACT_HEADERS
        ipsolver/two-level-solver.hpp)
  endif()

  convert_filenames_to_full_paths(CONTACT_SOURCES)
  convert_filenames_to_full_paths(CONTACT_HEADERS)

  set(CONTACT_COMMON_FILES
      EXTRA_SOURCES ${CONTACT_SOURCES}
      EXTRA_HEADERS ${CONTACT_HEADERS})

  add_custom_command(OUTPUT data_is_copied
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/meshes meshes
    COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
    COMMENT "Copying the miniapps/contact/meshes directory ...")
  add_custom_target(copy_contact_data DEPENDS data_is_copied)

  add_mfem_miniapp(contact
    MAIN contact.cpp
    ${CONTACT_COMMON_FILES}
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem)
  add_dependencies(contact copy_contact_data)

  add_mfem_miniapp(contact-vis
    MAIN contact-vis.cpp
    ${CONTACT_COMMON_FILES}
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common mfem)

# The way the libraries are included, by default, in the command line is based
# on the idea that MFEM depends on the other libraries. Here, the setting is
# such that Tribol, in fact, depends on MFEM, while MFEM does not depend on
# Tribol. This addresses that issue.
  target_link_libraries(contact PUBLIC ${TPL_LIBRARIES} mfem)
  target_link_libraries(contact-vis PUBLIC ${TPL_LIBRARIES} mfem)

  # Add parallel tests.
  if (MFEM_ENABLE_TESTING)
    add_test(NAME contact_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:contact>
      ${MPIEXEC_POSTFLAGS})

    if (MFEM_USE_MUMPS OR MFEM_USE_MKL_CPARDISO)
      add_test(NAME contact_AMGF_np=${MFEM_MPI_NP}
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:contact> -ls 6
        ${MPIEXEC_POSTFLAGS})
    endif()
  endif()

endif()