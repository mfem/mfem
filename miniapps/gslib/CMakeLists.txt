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

if (MFEM_USE_GSLIB)

  add_mfem_miniapp(field-diff
    MAIN field-diff.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(findpts
    MAIN findpts.cpp
    LIBRARIES mfem)

  add_mfem_miniapp(field-interp
    MAIN field-interp.cpp
    LIBRARIES mfem)

  # Parallel apps.
  if (MFEM_USE_MPI)
    add_mfem_miniapp(pfindpts
      MAIN pfindpts.cpp
      LIBRARIES mfem)
  endif()

endif()
