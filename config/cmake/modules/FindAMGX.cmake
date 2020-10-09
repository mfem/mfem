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

# Defines the following variables:
#   - AMGX_FOUND
#   - AMGX_LIBRARIES
#   - AMGX_INCLUDE_DIRS

include(MfemCmakeUtilities)
set(AMGX_REQUIRED_LIBRARIES cusparse cusolver cublas nvToolsExt)
mfem_find_package(AMGX AMGX AMGX_DIR "include" "amgx_c.h" "lib" "amgx"
  "Paths to headers required by AMGX." "Libraries required by AMGX.")
