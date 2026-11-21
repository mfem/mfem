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

add_subdirectory(structure)
add_subdirectory(ale)

# Headers are included relative to this directory, e.g.
# "structure/elasticity_solver.hpp" and "ale/mesh_laplacian.hpp".
set(MFEM_FSI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

# Coupled fluid-structure miniapps
foreach (fsi_app
    navier-fsi-two-domains
    navier-fsi-piston_separate
    navier-fsi-turek
    navier-fsi-turek_old
    navier-fsi-cylinder)
  add_mfem_miniapp(${fsi_app}
    MAIN ${fsi_app}.cpp
    EXTRA_SOURCES aitken.cpp
    EXTRA_HEADERS aitken.hpp
    LIBRARIES navier elasticity ale-deformation mfem-common-fsi)
  target_include_directories(${fsi_app} PRIVATE ${MFEM_FSI_INCLUDE_DIRS})
endforeach()

# Structure-only tests
foreach (csm_test
    test-csm-only
    test-csm-mms
    test-csm-robin-mms
    test-csm-mms-prescribedvel
    test-csm-convergence)
  add_mfem_miniapp(${csm_test}
    MAIN ${csm_test}.cpp
    LIBRARIES elasticity)
  target_include_directories(${csm_test} PRIVATE ${MFEM_FSI_INCLUDE_DIRS})
endforeach()

# Mesh deformation tests
add_mfem_miniapp(test-mesh-deformation
  MAIN test-mesh-deformation.cpp
  LIBRARIES elasticity ale-deformation mfem-common-fsi)
target_include_directories(test-mesh-deformation
  PRIVATE ${MFEM_FSI_INCLUDE_DIRS})

add_mfem_miniapp(test-local-stiffening
  MAIN test-local-stiffening.cpp
  LIBRARIES ale-deformation mfem-common-fsi)
target_include_directories(test-local-stiffening
  PRIVATE ${MFEM_FSI_INCLUDE_DIRS})
