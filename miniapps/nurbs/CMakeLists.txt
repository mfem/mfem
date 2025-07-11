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

set(MESH_FILES
  cube-nurbs.mesh
  ijk-hex-nurbs.mesh
  plus-nurbs-permuted.mesh
  plus-nurbs.mesh
  square-nurbs.mesh
  two-cubes-nurbs-autoedge.mesh
  two-cubes-nurbs-rot.mesh
  two-cubes-nurbs.mesh
  two-squares-nurbs-autoedge.mesh
  two-squares-nurbs-rot.mesh
  two-squares-nurbs.mesh
)
# Add a target to copy the mesh files from the source directory; used by sample
# runs.
set(SRC_MESH_FILES)
foreach(MESH_FILE ${MESH_FILES})
  list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/meshes/${MESH_FILE})
endforeach()
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E make_directory meshes
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} meshes/
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying nurbs miniapps data files ...")
add_custom_target(copy_miniapps_nurbs_data DEPENDS data_is_copied)

add_mfem_miniapp(nurbs_ex1
  MAIN nurbs_ex1.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_ex1 copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_ex3
  MAIN nurbs_ex3.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_ex3 copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_ex5
  MAIN nurbs_ex5.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_ex5 copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_ex24
  MAIN nurbs_ex24.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_ex24 copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_curveint
  MAIN nurbs_curveint.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_curveint copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_naca_cmesh
  MAIN nurbs_naca_cmesh.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_naca_cmesh copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_printfunc
  MAIN nurbs_printfunc.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_printfunc copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_patch_ex1
  MAIN nurbs_patch_ex1.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_patch_ex1 copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_solenoidal
  MAIN nurbs_solenoidal.cpp
  LIBRARIES mfem)
  add_dependencies(nurbs_solenoidal copy_miniapps_nurbs_data)

add_mfem_miniapp(nurbs_surface
  MAIN nurbs_surface.cpp
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME nurbs_ex1_1d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2 -lod 3)

  add_test(NAME nurbs_ex1_1d_r1_o2_wbc_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/segment-nurbs.mesh -r 1 -o 2 -wbc -lod 3)

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

  add_test(NAME nurbs_ex1_neu_r2_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -r 2 --neu "3")

  add_test(NAME nurbs_ex1_weak_mp_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/ball-nurbs.mesh -o 2 --weak-bc -r 0)

  add_test(NAME nurbs_ex1_weak_patch_format_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh
    -o 2 --weak-bc -r 0)

  add_test(NAME nurbs_ex1_weak_patch_format_r1_ser
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh
    -o 2 --weak-bc -r 1)

  add_test(NAME nurbs_printfunc
    COMMAND $<TARGET_FILE:nurbs_printfunc>)

  if (MFEM_USE_LAPACK)
    add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_ser
      COMMAND $<TARGET_FILE:nurbs_patch_ex1> -no-vis -incdeg 3 -ref 2 -iro 8
      -patcha)
  endif()

  add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_pa_ser
    COMMAND $<TARGET_FILE:nurbs_patch_ex1> -no-vis -incdeg 3 -ref 2 -iro 8
    -patcha -pa)

  add_test(NAME nurbs_patch_ex1_o4_r2_iro8_patcha_fint_ser
    COMMAND $<TARGET_FILE:nurbs_patch_ex1> -no-vis -incdeg 3 -ref 2 -iro 8
    -patcha -fint)

  add_test(NAME nurbs_curveint_unit_weight_ser
    COMMAND $<TARGET_FILE:nurbs_curveint> -no-vis -no-visit
    -uw -n 9)

  add_test(NAME nurbs_curveint_non_unit_weight_ser
    COMMAND $<TARGET_FILE:nurbs_curveint> -no-vis -no-visit
    -nw -n 9)

  add_test(NAME nurbs_naca_cmesh_ser
    COMMAND $<TARGET_FILE:nurbs_naca_cmesh> -no-vis -no-visit -ntail 80 -nbnd 80
    -ntip 20 -nwake 40 -sw 2.0 -sbnd 2.5 -stip 1.1 -aoa 3)

  add_test(NAME nurbs_ex1_two_squares_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs.mesh -o 1
    -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_squares_rot_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs-rot.mesh
    -o 1 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_squares_autoedge_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares-nurbs-autoedge.mesh
    -o 1 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-squares.ref)

  add_test(NAME nurbs_ex1_two_cubes_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs.mesh -o 1
    -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)

  add_test(NAME nurbs_ex1_two_cubes_rot_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs-rot.mesh -o 1
    -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)

  add_test(NAME nurbs_ex1_two_cubes_autoedge_knot_insert
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes-nurbs-autoedge.mesh
    -o 1 -r 3 -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/two-cubes.ref)

  add_test(NAME nurbs_ex1_periodic_2d
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -o 2 -r 2 --master "3"
    --slave "4")

  add_test(NAME nurbs_ex1_periodic_3d
    COMMAND $<TARGET_FILE:nurbs_ex1> -no-vis
    -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/cube-nurbs.mesh
    -rf ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/cube.ref
    --master "1" --slave "2")

  add_test(NAME nurbs_ex3_2d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex3> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-nurbs.mesh -r 1 -o 2)

  add_test(NAME nurbs_ex3_3d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex3> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2)

  add_test(NAME nurbs_ex5_2d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex5> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-nurbs.mesh -r 1 -o 2)

  add_test(NAME nurbs_ex5_3d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_ex5> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2)

  add_test(NAME nurbs_ex24_2d_r1_o2_p0_ser
    COMMAND $<TARGET_FILE:nurbs_ex24> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -r 1 -o 2 -p 0)

  add_test(NAME nurbs_ex24_2d_r1_o2_p2_ser
    COMMAND $<TARGET_FILE:nurbs_ex24> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -r 1 -o 2 -p 2)

  add_test(NAME nurbs_ex24_3d_r1_o2_p0_ser
    COMMAND $<TARGET_FILE:nurbs_ex24> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2 -p 0)

  add_test(NAME nurbs_ex24_3d_r1_o2_p1_ser
    COMMAND $<TARGET_FILE:nurbs_ex24> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2 -p 1)

  add_test(NAME nurbs_ex24_3d_r1_o2_p2_ser
    COMMAND $<TARGET_FILE:nurbs_ex24> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2 -p 2)

  add_test(NAME nurbs_solenoidal_2d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_solenoidal> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/pipe-nurbs-2d.mesh -r 1 -o 2)

  add_test(NAME nurbs_solenoidal_3d_r1_o2_ser
    COMMAND $<TARGET_FILE:nurbs_solenoidal> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/cube-nurbs.mesh -r 1 -o 2)

  add_test(NAME nurbs_surface_10_10_10_10_ex1_o3_ser
    COMMAND $<TARGET_FILE:nurbs_surface> -no-vis
        -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 1 -orig)

  add_test(NAME nurbs_surface_10_10_40_40_ex1_o3_ser
    COMMAND $<TARGET_FILE:nurbs_surface> -no-vis
        -o 3 -nx 10 -ny 10 -fnx 40 -fny 14 -ex 1)

endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(nurbs_ex1p
    MAIN nurbs_ex1p.cpp
    LIBRARIES mfem)
    add_dependencies(nurbs_ex1p copy_miniapps_nurbs_data)

  add_mfem_miniapp(nurbs_ex11p
    MAIN nurbs_ex11p.cpp
    LIBRARIES mfem)
    add_dependencies(nurbs_ex11p copy_miniapps_nurbs_data)

  if (MFEM_ENABLE_TESTING)
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

    add_test(NAME nurbs_ex1p_periodic_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex1p> -no-vis
      -m ${PROJECT_SOURCE_DIR}/miniapps/nurbs/meshes/square-nurbs.mesh
      -r 4  -pm "1" -ps "2"
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME nurbs_ex1_weak_patch_format_np=4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1p> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2
        --weak-bc -r 0)

    add_test(NAME nurbs_ex1_weak_patch_format_r1_np=4
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
        ${MPIEXEC_PREFLAGS}  $<TARGET_FILE:nurbs_ex1p> -no-vis
        -m ${PROJECT_SOURCE_DIR}/data/square-disc-nurbs-patch.mesh -o 2
        --weak-bc -r 1)

    add_test(NAME nurbs_ex11p_np=4
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:nurbs_ex11p> -no-vis
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
