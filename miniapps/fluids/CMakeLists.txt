# Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

if (MFEM_USE_MPI)
  add_mfem_miniapp(navier_tgv2d
    MAIN navier_tgv2d.cpp
    LIBRARIES mfem navier)

  add_mfem_miniapp(navier_tgv3d
    MAIN navier_tgv3d.cpp
    LIBRARIES mfem navier)

  add_mfem_miniapp(navier_kovasznay
    MAIN navier_kovasznay.cpp
    LIBRARIES mfem navier)

  add_mfem_miniapp(navier_mms
    MAIN navier_mms.cpp
    LIBRARIES mfem navier)

  add_library(navier
    navier_solver.cpp navier_solver.hpp
    ortho_solver.cpp ortho_solver.hpp)

  target_link_libraries(navier mfem)

  target_include_directories(navier INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/navier>)

  install(TARGETS navier
    EXPORT ${PROJECT_NAME_UC}Targets
    DESTINATION ${INSTALL_LIB_DIR})

  install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/navier_solver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ortho_solver.hpp
    DESTINATION include/navier)
endif()
