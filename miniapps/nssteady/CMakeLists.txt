# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
  list(APPEND SNAVIER_COMMON_SOURCES
      snavier_picard_monolithic.cpp
      snavier_newton_monolithic.cpp
      snavier_picard_segregated.cpp
      snavier_dg.cpp
      custom_bilinteg.cpp)
  list(APPEND SNAVIER_COMMON_HEADERS
      snavier_picard_monolithic.hpp
      snavier_newton_monolithic.hpp
      snavier_picard_segregated.hpp
      snavier_dg.hpp
      custom_bilinteg.hpp)

  convert_filenames_to_full_paths(SNAVIER_COMMON_SOURCES)
  convert_filenames_to_full_paths(SNAVIER_COMMON_HEADERS)

  set(SNAVIER_COMMON_FILES
      EXTRA_SOURCES ${SNAVIER_COMMON_SOURCES}
      EXTRA_HEADERS ${SNAVIER_COMMON_HEADERS})


  add_mfem_miniapp(test_snavier_dg
    MAIN test_snavier_dg.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(test_integrator
    MAIN test_integrator.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(test_monolithic_stokes
    MAIN MMS_monolithic_stokes.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(MMS_monolithic_picard_nssteady
    MAIN MMS_monolithic_picard_nssteady.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(MMS_solver_monolithic_picard_nssteady
    MAIN MMS_solver_monolithic_picard_nssteady.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(MMS_solver_monolithic_newton_nssteady
    MAIN MMS_solver_monolithic_newton_nssteady.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(LidCavity_solver_monolithic_picard_nssteady
    MAIN LidCavity_solver_monolithic_picard_nssteady.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(FlowCyl_solver_monolithic_picard_nssteady
    MAIN FlowCyl_solver_monolithic_picard_nssteady.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(MMS_segregated_picard
    MAIN MMS_segregated_picard.cpp
    ${SNAVIER_COMMON_FILES}
    LIBRARIES mfem)

# Create a symbolic link for meshes
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
               ${CMAKE_CURRENT_SOURCE_DIR}/Mesh ${CMAKE_CURRENT_BINARY_DIR}/Mesh
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

endif ()
