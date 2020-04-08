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

if (MFEM_USE_GSLIB)

  add_mfem_miniapp(field-diff
    MAIN field-diff.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(findpts
    MAIN findpts.cpp
    LIBRARIES mfem)

  # Parallel apps.
  if (MFEM_USE_MPI)
    add_mfem_miniapp(pfindpts
      MAIN findpts.cpp
      LIBRARIES mfem)
  endif()

endif()
