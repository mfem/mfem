# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
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
  add_mfem_miniapp(flow_tgv2d
    MAIN flow.cpp
    EXTRA_SOURCES flow_solver.cpp ortho_solver.cpp
    EXTRA_HEADERS flow_solver.hpp ortho_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_foc
    MAIN flow_foc.cpp
    EXTRA_SOURCES flow_solver.cpp ortho_solver.cpp
    EXTRA_HEADERS flow_solver.hpp ortho_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_foc3d
    MAIN flow_foc3d.cpp
    EXTRA_SOURCES flow_solver.cpp ortho_solver.cpp
    EXTRA_HEADERS flow_solver.hpp ortho_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_instest
    MAIN ins_transient_projection.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_kovasznay
    MAIN flow_kovasznay.cpp
    EXTRA_SOURCES flow_solver.cpp ortho_solver.cpp
    EXTRA_HEADERS flow_solver.hpp ortho_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_tgv3d
    MAIN flow_tgv.cpp
    EXTRA_SOURCES flow_solver.cpp ortho_solver.cpp
    EXTRA_HEADERS flow_solver.hpp ortho_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(flow_poisson
    MAIN poisson.cpp
    LIBRARIES mfem)

  add_library(flow flow_solver.cpp flow_solver.hpp ortho_solver.cpp ortho_solver.hpp)
  target_link_libraries(flow mfem)
  add_custom_command(TARGET flow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/flow_solver.hpp $<TARGET_FILE_DIR:flow>/flow_solver.hpp)

  # Add the corresponding tests to the "test" target
  add_test(NAME flow_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:flow>
    ${MPIEXEC_POSTFLAGS})
endif()
