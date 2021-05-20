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

if (MFEM_USE_MPI)
  add_mfem_miniapp(stix1d
    MAIN stix1d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  add_mfem_miniapp(stix2d
    MAIN stix2d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  add_mfem_miniapp(stix3d
    MAIN stix3d.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  add_mfem_miniapp(stix1d_dh
    MAIN stix1d_dh.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp cold_plasma_dielectric_dh_solver.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp cold_plasma_dielectric_dh_solver.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

  add_mfem_miniapp(stix2d_dh
    MAIN stix2d_dh.cpp
    EXTRA_SOURCES cold_plasma_dielectric_solver.cpp cold_plasma_dielectric_coefs.cpp cold_plasma_dielectric_dh_solver.cpp
    EXTRA_HEADERS cold_plasma_dielectric_solver.hpp cold_plasma_dielectric_coefs.hpp cold_plasma_dielectric_dh_solver.hpp plasma.hpp ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem_miniapps_common)

endif()
