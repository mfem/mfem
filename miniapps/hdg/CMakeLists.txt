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

list(APPEND HDG_COMMON_SOURCES
  darcyop.cpp)

list(APPEND HDG_COMMON_HEADERS
  darcyop.hpp)

convert_filenames_to_full_paths(HDG_COMMON_SOURCES)
convert_filenames_to_full_paths(HDG_COMMON_HEADERS)

set(HDG_COMMON_FILES
    EXTRA_SOURCES ${HDG_COMMON_SOURCES}
    EXTRA_HEADERS ${HDG_COMMON_HEADERS})

add_mfem_miniapp(anisodiff
    MAIN anisodiff.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

add_mfem_miniapp(anisodiff-hr
    MAIN anisodiff-hr.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

add_mfem_miniapp(convdiff
    MAIN convdiff.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

add_mfem_miniapp(mfem-logo
    MAIN mfem-logo.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

# navierstokes takes its globalisation from KINSolver's KIN_LINESEARCH rather
# than carrying a backtracking Newton of its own, so it requires SUNDIALS and
# is not built without it. The GNU makefile drops it from SEQ_MINIAPPS for the
# same reason.
if (MFEM_USE_SUNDIALS)
  add_mfem_miniapp(navierstokes
      MAIN navierstokes.cpp
      ${HDG_COMMON_FILES}
      LIBRARIES mfem)
endif()

if (MFEM_USE_MPI)
  add_mfem_miniapp(panisodiff
    MAIN panisodiff.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

  add_mfem_miniapp(pconvdiff
    MAIN pconvdiff.cpp
    ${HDG_COMMON_FILES}
    LIBRARIES mfem)

  # pnavierstokes carries navierstokes's SUNDIALS guard, so it is doubly
  # conditional: MPI for the parallel spaces, SUNDIALS for KINSOL. The GNU
  # makefile drops it from PAR_MINIAPPS for the same reason.
  if (MFEM_USE_SUNDIALS)
    add_mfem_miniapp(pnavierstokes
      MAIN pnavierstokes.cpp
      ${HDG_COMMON_FILES}
      LIBRARIES mfem)
  endif()
endif()
