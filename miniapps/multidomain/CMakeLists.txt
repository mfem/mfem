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
  multidomain-hex.mesh
)

# Add a target to copy the mesh files from the source directory; used by sample
# runs.
set(SRC_MESH_FILES)
foreach(MESH_FILE ${MESH_FILES})
  list(APPEND SRC_MESH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${MESH_FILE})
endforeach()
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_MESH_FILES} .
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying multidomain miniapps data files ...")
add_custom_target(copy_miniapps_multidomain_data DEPENDS data_is_copied)

# Parallel apps.
if (MFEM_USE_MPI)
  add_mfem_miniapp(multidomain
    MAIN multidomain.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem-common)
  add_dependencies(multidomain copy_miniapps_multidomain_data)

  add_mfem_miniapp(multidomain_nd
    MAIN multidomain_nd.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)
  add_dependencies(multidomain_nd copy_miniapps_multidomain_data)

  add_mfem_miniapp(multidomain_rt
    MAIN multidomain_rt.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)
  add_dependencies(multidomain_rt copy_miniapps_multidomain_data)

# Add parallel tests.
  if (MFEM_ENABLE_TESTING)
    add_test(NAME multidomain_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:multidomain> -no-vis -tf 0.001
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME multidomain_nd_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:multidomain_nd> -no-vis -tf 0.001
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME multidomain_rt_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:multidomain_rt> -no-vis -tf 0.001
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
