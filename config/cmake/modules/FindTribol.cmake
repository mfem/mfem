# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#   - TRIBOL_FOUND
#   - TRIBOL_LIBRARIES
#   - TRIBOL_INCLUDE_DIRS

include(MfemCmakeUtilities)
# Note: components are enabled based on the find_package() parameters.
mfem_find_package(Tribol TRIBOL TRIBOL_DIR "include" tribol/config.hpp "lib" tribol
  "Paths to headers required by Tribol." "Libraries required by Tribol."
  ADD_COMPONENT redecomp
    "include" redecomp/redecomp.hpp "lib" redecomp)
