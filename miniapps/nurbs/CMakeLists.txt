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

add_mfem_miniapp(nurbs_ex1
  MAIN nurbs_ex1.cpp
  LIBRARIES mfem)

add_test(NAME nurbs_ex1_r0_o4_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis -r 0 -o 4)

add_test(NAME nurbs_ex1_r2_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis -r 2)

add_test(NAME nurbs_ex1_per_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/beam-hex-nurbs.mesh -pm 1 -ps 2)

add_test(NAME nurbs_ex1_lap_r0_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 0)

add_test(NAME nurbs_ex1_lap_r2_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 2)

add_test(NAME nurbs_ex1_weak_r0_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 0)

add_test(NAME nurbs_ex1_weak_r2_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 2)

add_test(NAME nurbs_ex1_weak_mp_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/ball-nurbs.mesh -o 2 --weak-bc -r 0)

add_test(NAME nurbs_ex1_weak_patch_format_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0)

add_test(NAME nurbs_ex1_weak_patch_format_r1_ser
  COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
  -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1)

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
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis -m 
    ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -no-ibp
    ${MPIEXEC_POSTFLAGS})

  add_test(NAME nurbs_ex1p_weak_mp_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/ball-nurbs.mesh -o 2 --weak-bc -r 0
    ${MPIEXEC_POSTFLAGS})

add_test(NAME nurbs_ex1_weak_patch_format_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0)

add_test(NAME nurbs_ex1_weak_patch_format_r1_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1)

  add_mfem_miniapp(nurbs_ex11p
    MAIN nurbs_ex11p.cpp
    LIBRARIES mfem)

  add_test(NAME nurbs_ex11p_np=4
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex11p> -no-vis
    ${MPIEXEC_POSTFLAGS})
endif()
