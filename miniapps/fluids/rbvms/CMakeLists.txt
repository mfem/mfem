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


##target_link_libraries(rbvms PRIVATE ${LIBRARIES} util ${CMAKE_DL_LIBS})

add_mfem_miniapp(rbvms
  MAIN rbvms.cpp 
  EXTRA_SOURCES integrator.cpp
  EXTRA_HEADERS integrator.hpp
  LIBRARIES mfem)

LIST(APPEND RBVMS_FILES
        README.md
        von-karman-nurbs.mesh
        von-karman-tri.geo)
foreach(RBVMS_FILE ${RBVMS_FILES})
   add_custom_command(TARGET rbvms POST_BUILD #OUTPUT ${RBVMS_FILE}
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${CMAKE_CURRENT_SOURCE_DIR}/${RBVMS_FILE} ${RBVMS_FILE}
                      COMMENT "copy: ${RBVMS_FILE}")
endforeach()

if (MFEM_ENABLE_TESTING)
   add_test(NAME rbvms-ldc
         COMMAND  $<TARGET_FILE:rbvms> -m ../../../data/square-nurbs.mesh -r 1 -o 2
                  -p 0 -rho 1 -mu 0.001 --strong-bdr 1 -s 21 -dt 0.01 -tf 1)
   add_test(NAME rbvms-vkvs
         COMMAND  $<TARGET_FILE:rbvms> -m von-karman-nurbs.mesh  -r 1 -o 2
                  -p 1 --strong-bdr "1 2" --outflow-bdr 4 --weak-bdr 3 
                  -rho 1 -mu 0.01 -s 45 -dt 0.1  -tf 1)
endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(prbvms
  MAIN prbvms.cpp
  EXTRA_SOURCES integrator.cpp
  EXTRA_HEADERS integrator.hpp
  LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME prbvms_ldc_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:prbvms> -m ../../../data/square-nurbs.mesh -o 2 -r 3
               -p 0 -rho 1 -mu 0.001 --strong-bdr 1 -s 21 -dt 0.01 -tf 1
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME prbvms_vkvs_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:prbvms> -m von-karman-nurbs.mesh  -r 1 -o 2
                  -p 1 --strong-bdr "1 2" --outflow-bdr 4 --weak-bdr 3 
                  -rho 1 -mu 0.01 -s 45 -dt 0.1  -tf 1
      ${MPIEXEC_POSTFLAGS})
  endif()
  
  
endif()




