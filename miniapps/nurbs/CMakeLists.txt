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

add_mfem_miniapp(nurbs_ex1
  MAIN ex1.cpp
  LIBRARIES mfem)

add_test(NAME nurbs_ex1_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis)

add_test(NAME nurbs_ex1_per_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2)

if (MFEM_USE_MPI)
  add_mfem_miniapp(nurbs_ex1p
    MAIN ex1p.cpp
    LIBRARIES mfem)

  add_test(NAME nurbs_ex1p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
    ${MPIEXEC_POSTFLAGS})

  add_mfem_miniapp(nurbs_ex11p
    MAIN ex11p.cpp
    LIBRARIES mfem)

  add_test(NAME nurbs_ex11p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex11p> -no-vis
    ${MPIEXEC_POSTFLAGS})
endif()
