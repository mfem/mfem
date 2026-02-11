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

if (MFEM_USE_MPI)
  list(APPEND PLASMA_COMMON_SOURCES)

  list(APPEND PLASMA_COMMON_HEADERS
  plasma.hpp)

  convert_filenames_to_full_paths(PLASMA_COMMON_SOURCES)
  convert_filenames_to_full_paths(PLASMA_COMMON_HEADERS)

  set(PLASMA_COMMON_FILES
      EXTRA_SOURCES ${PLASMA_COMMON_SOURCES}
      EXTRA_HEADERS ${PLASMA_COMMON_HEADERS})

endif()
