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

# Sets the following variables:
#   - MUMPS_FOUND
#   - MUMPS_INCLUDE_DIRS
#   - MUMPS_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(MUMPS MUMPS MUMPS_DIR
  "include" dmumps_c.h "lib" dmumps
  "Paths to headers required by MUMPS."
  "Libraries required by MUMPS."
  ADD_COMPONENT mumps_common "include" dmumps_c.h "lib" mumps_common
  ADD_COMPONENT pord "include" dmumps_c.h "lib" pord)
