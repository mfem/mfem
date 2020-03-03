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

# Find the libraries needed for backtrace in MFEM. This is a "meta-package",
# i.e. it simply combines all packages from MFEMBacktrace_REQUIRED_PACKAGES.
# Defines the following variables:
#   - MFEMBacktrace_FOUND
#   - MFEMBacktrace_LIBRARIES    (if needed)
#   - MFEMBacktrace_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
mfem_find_package(MFEMBacktrace MFEMBacktrace MFEMBacktrace_DIR "" "" "" ""
  "Paths to headers required by MFEM backtrace."
  "Libraries required by MFEM backtrace.")
