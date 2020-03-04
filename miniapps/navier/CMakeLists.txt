# Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at the
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
  set(NAVIER_COMMON_FILES
      navier_solver.cpp
      navier_solver.hpp
      ortho_solver.cpp
      ortho_solver.hpp
  )

add_mfem_miniapp(navier_mms
    MAIN navier_mms.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_kovasznay
    MAIN navier_kovasznay.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_tgv
    MAIN navier_tgv.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(navier_shear
    MAIN navier_shear.cpp
    ${NAVIER_COMMON_FILES}
    LIBRARIES mfem)

  # add_mfem_miniapp(navier_3dfoc
  #   MAIN navier_3dfoc.cpp
  #   ${NAVIER_COMMON_FILES}
  #   LIBRARIES mfem)
endif ()