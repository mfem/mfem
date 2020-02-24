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

configure_file(${PROJECT_SOURCE_DIR}/miniapps/nurbs/square-nurbs.mesh
               ${PROJECT_BINARY_DIR}/miniapps/nurbs/square-nurbs.mesh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/nurbs/cube-nurbs.mesh
               ${PROJECT_BINARY_DIR}/miniapps/nurbs/cube-nurbs.mesh
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/nurbs/pipe-nurbs-2d.mesh
               ${PROJECT_BINARY_DIR}/miniapps/nurbs/pipe-nurbs-2d.mesh
               COPYONLY)

add_mfem_miniapp(nurbs_ex1
  MAIN nurbs_ex1.cpp
  LIBRARIES mfem)

add_test(NAME nurbs_ex1_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis)

add_test(NAME nurbs_ex1_per_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2)

add_test(NAME nurbs_ex1_lap_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m pipe-nurbs-2d.mesh -o 2 -no-ibp)


if (MFEM_USE_MPI)
  add_mfem_miniapp(nurbs_ex1p
    MAIN nurbs_ex1p.cpp
    LIBRARIES mfem)

  add_test(NAME nurbs_ex1p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
    ${MPIEXEC_POSTFLAGS})

  add_test(NAME nurbs_ex1p_lap_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis -m pipe-nurbs-2d.mesh -o 2 -no-ibp
    ${MPIEXEC_POSTFLAGS})

  add_mfem_miniapp(nurbs_ex11p
    MAIN nurbs_ex11p.cpp
    LIBRARIES mfem)

  add_test(NAME nurbs_ex11p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex11p> -no-vis
    ${MPIEXEC_POSTFLAGS})
endif()
