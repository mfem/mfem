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
