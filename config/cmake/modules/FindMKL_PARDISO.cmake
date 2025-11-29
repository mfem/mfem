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
#   - MKL_PARDISO_FOUND
#   - MKL_PARDISO_LIBRARIES
#   - MKL_PARDISO_INCLUDE_DIRS

if(NOT MKL_LIBRARY_DIR)
  message(WARNING "Using default MKL library path. Double check the variable MKL_LIBRARY_DIR")
  set(MKL_LIBRARY_DIR "lib/intel64")
endif()

include(MfemCmakeUtilities)
mfem_find_package(MKL_PARDISO MKL_PARDISO
    MKL_PARDISO_DIR "include" mkl_pardiso.h ${MKL_LIBRARY_DIR} mkl_core
  "Paths to headers required by MKL Pardiso." "Libraries required by MKL PARDISO."
  ADD_COMPONENT MKL_LP64 "include" "" ${MKL_LIBRARY_DIR} mkl_intel_lp64
  ADD_COMPONENT MKL_SEQUENTIAL "include" "" ${MKL_LIBRARY_DIR} mkl_sequential)
