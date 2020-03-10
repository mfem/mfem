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
#   - GECKO_FOUND
#   - GECKO_LIBRARIES
#   - GECKO_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(Gecko GECKO GECKO_DIR "include;inc" graph.h "lib" gecko
  "Paths to headers required by Gecko." "Libraries required by Gecko.")
