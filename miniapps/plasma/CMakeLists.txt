// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

if (MFEM_USE_MPI)
  add_mfem_miniapp(stix1d
    MAIN stix1d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

  add_mfem_miniapp(stix2d
    MAIN stix2d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

  add_mfem_miniapp(stix3d
    MAIN stix3d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

  add_mfem_miniapp(braginskii
    MAIN braginskii.cpp
    EXTRA_SOURCES braginskii_solver.cpp braginskii_coefs.cpp
    EXTRA_HEADERS braginskii_solver.hpp braginskii_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

  add_mfem_miniapp(transport2d
    MAIN transport2d.cpp
    EXTRA_SOURCES transport_solver.cpp plasma.cpp
    EXTRA_HEADERS transport_solver.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

endif()
