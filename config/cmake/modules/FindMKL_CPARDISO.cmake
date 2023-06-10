# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#   - MKL_CPARDISO_FOUND
#   - MKL_CPARDISO_LIBRARIES
#   - MKL_CPARDISO_INCLUDE_DIRS

# It also creates the target (CMake package style) MKL_CPARDISO::MKL_CPARDISO

if(NOT MKL_MPI_WRAPPER_LIB)
  message(FATAL_ERROR "MKL CPardiso enabled but no MKL MPI Wrapper lib specified")
endif()

if(NOT MKL_LIBRARY_DIR)
  message(WARNING "Using default MKL library path. Double check the variable MKL_LIBRARY_DIR")
  set(MKL_LIBRARY_DIR "lib")
endif()

include(MfemCmakeUtilities)
mfem_find_package(MKL_CPARDISO MKL_CPARDISO
    MKL_CPARDISO_DIR "include" mkl_cluster_sparse_solver.h ${MKL_LIBRARY_DIR} mkl_core
  "Paths to headers required by MKL CPardiso." "Libraries required by MKL CPARDISO."
  ADD_COMPONENT MKL_LP64 "include" "" ${MKL_LIBRARY_DIR} mkl_intel_lp64
  ADD_COMPONENT MKL_SEQUENTIAL "include" "" ${MKL_LIBRARY_DIR} mkl_sequential
  ADD_COMPONENT MKL_MPI_WRAPPER "include" "" ${MKL_LIBRARY_DIR} ${MKL_MPI_WRAPPER_LIB})

if(MKL_CPARDISO_FOUND)
  mfem_library_to_package(MKL_CPARDISO::MKL_CPARDISO "${MKL_CPARDISO_INCLUDE_DIRS}" "${MKL_CPARDISO_LIBRARIES}")
endif()
