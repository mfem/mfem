# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(opt_test
  MAIN opt_test.cpp
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME optimization_test COMMAND opt_test)
  add_test(NAME optimization_test_debug_device COMMAND opt_test -d debug)
endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(opt_examples
    MAIN opt_examples.cpp
    LIBRARIES mfem)

  if (MFEM_ENABLE_TESTING)
    add_test(NAME optimization_test_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:opt_test> ${MPIEXEC_POSTFLAGS})
    add_test(NAME optimization_examples_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:opt_examples> ${MPIEXEC_POSTFLAGS})
  endif()
endif()
