# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_PARELAG AND MFEM_USE_MPI)

  configure_file(${PROJECT_SOURCE_DIR}/miniapps/parelag/MultilevelHcurlSolver_example_parameters.xml
                 ${PROJECT_BINARY_DIR}/miniapps/parelag/MultilevelHcurlSolver_example_parameters.xml
                 COPYONLY)

  configure_file(${PROJECT_SOURCE_DIR}/miniapps/parelag/MultilevelHdivSolver_example_parameters.xml
                 ${PROJECT_BINARY_DIR}/miniapps/parelag/MultilevelHdivSolver_example_parameters.xml
                 COPYONLY)

  add_mfem_miniapp(MultilevelHcurlSolver
    MAIN MultilevelHcurlSolver.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(MultilevelHdivSolver
    MAIN MultilevelHdivSolver.cpp
    LIBRARIES mfem)

# XXX: The way the libraries are included, by default, in the command line is
#      based on the idea that MFEM depends on the other libraries. Here, the
#      setting is such that ParELAG, in fact, depends on MFEM, while MFEM does
#      not depend on ParELAG. This addresses that issue.
  if (CMAKE_VERSION VERSION_GREATER 2.8.11)
    target_link_libraries(MultilevelHcurlSolver PUBLIC ${TPL_LIBRARIES} mfem)
    target_link_libraries(MultilevelHdivSolver PUBLIC ${TPL_LIBRARIES} mfem)
  else()
    target_link_libraries(MultilevelHcurlSolver ${TPL_LIBRARIES} mfem)
    target_link_libraries(MultilevelHdivSolver ${TPL_LIBRARIES} mfem)
  endif()

endif()
