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

if (MFEM_USE_MPI AND MFEM_USE_GSLIB)
  add_mfem_miniapp(electrostatic-pic
    MAIN electrostatic-pic.cpp
    EXTRA_HEADERS ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME electrostatic-pic_np=${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:electrostatic-pic> -rdf 1 -npt 102400 -k 0.2855993321 -a 0.05 -nt 20 -nx 16 -ny 16 -O 1 -q 0.001181640625 -m 0.001181640625 -ocf 1000 -dt 0.1
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
