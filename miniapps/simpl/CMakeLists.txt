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
  add_mfem_miniapp(simpl
    MAIN simpl.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp logger.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp logger.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(test_proj
    MAIN test_proj.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp logger.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp logger.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(oc
    MAIN oc.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp logger.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp logger.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(oc_org
    MAIN oc_org.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp logger.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp logger.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(mma
    MAIN mma.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp logger.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp logger.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(filter_test
    MAIN filter_test.cpp
    EXTRA_SOURCES linear_solver.cpp topopt.cpp topopt_problems.cpp funs.cpp
    EXTRA_HEADERS linear_solver.hpp topopt.hpp topopt_problems.hpp funs.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(diffusion_test
    MAIN diffusion_test.cpp
    EXTRA_SOURCES linear_solver.cpp
    EXTRA_HEADERS linear_solver.hpp
    LIBRARIES mfem)
  add_mfem_miniapp(elasticity_test
    MAIN elasticity_test.cpp
    EXTRA_SOURCES linear_solver.cpp
    EXTRA_HEADERS linear_solver.hpp
    LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME simpl_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:simpl> -rs 4 -rp 0
      ${MPIEXEC_POSTFLAGS})

  endif()
endif()
