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

# See the file INSTALL for description of the configuration options.

# Default options. To replace these, copy this file to user.cmake and modify it.

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
endif()

# MFEM options. Set to mimic the default "default.mk" file.
option(MFEM_USE_MPI "Enable MPI parallel build" OFF)
option(MFEM_USE_GZSTREAM "Enable gzstream for compressed data streams." OFF)
option(MFEM_USE_LIBUNWIND "Enable backtrace for errors." OFF)
option(MFEM_USE_LAPACK "Enable LAPACK usage" OFF)
option(MFEM_THREAD_SAFE "Enable thread safety" OFF)
option(MFEM_USE_OPENMP "Enable OpenMP usage" OFF)
option(MFEM_USE_MEMALLOC "Enable the internal MEMALLOC option." ON)
option(MFEM_USE_SUNDIALS "Enable SUNDIALS usage" OFF)
option(MFEM_USE_MESQUITE "Enable MESQUITE usage" OFF)
option(MFEM_USE_SUITESPARSE "Enable SuiteSparse usage" OFF)
option(MFEM_USE_SUPERLU "Enable SuperLU_DIST usage" OFF)
option(MFEM_USE_STRUMPACK "Enable STRUMPACK usage" OFF)
option(MFEM_USE_GECKO "Enable GECKO usage" OFF)
option(MFEM_USE_GNUTLS "Enable GNUTLS usage" OFF)
option(MFEM_USE_NETCDF "Enable NETCDF usage" OFF)
option(MFEM_USE_PETSC "Enable PETSc support." OFF)
option(MFEM_USE_MPFR "Enable MPFR usage." OFF)
option(MFEM_USE_SIDRE "Enable Axom/Sidre usage" OFF)

# Allow a user to disable testing, examples, and/or miniapps at CONFIGURE TIME
# if they don't want/need them (e.g. if MFEM is "just a dependency" and all they
# need is the library, building all that stuff adds unnecessary overhead). To
# match "makefile" behavior, they are all enabled by default.
option(MFEM_ENABLE_TESTING "Enable the ctest framework for testing" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" OFF)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)

# Setting CXX/MPICXX on the command line or in user.cmake will overwrite the
# autodetected C++ compiler.
# set(CXX g++)
# set(MPICXX mpicxx)

set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The *_DIR paths below will be the first place searched for the corresponding
# headers and library. If these fail, then standard cmake search is performed.
# Note: if the variables are already in the cache, they are not overwritten.

set(HYPRE_DIR "${MFEM_DIR}/../hypre-2.10.0b/src/hypre" CACHE PATH
    "Path to the hypre library.")
# If hypre was compiled to depend on BLAS and LAPACK:
# set(HYPRE_REQUIRED_PACKAGES "BLAS" "LAPACK" CACHE STRING
#     "Packages that HYPRE depends on.")

set(METIS_DIR "${MFEM_DIR}/../metis-4.0" CACHE PATH "Path to the METIS library.")

set(LIBUNWIND_DIR "" CACHE PATH "Path to Libunwind.")

set(SUNDIALS_DIR "${MFEM_DIR}/../sundials-2.7.0" CACHE PATH
    "Path to the SUNDIALS library.")
# The following may be necessary, if SUNDIALS was built with KLU:
# set(SUNDIALS_REQUIRED_PACKAGES "SuiteSparse/KLU/AMD/BTF/COLAMD/config"
#     CACHE STRING "Additional packages required by SUNDIALS.")

set(MESQUITE_DIR "${MFEM_DIR}/../mesquite-2.99" CACHE PATH
    "Path to the Mesquite library.")

set(SuiteSparse_DIR "${MFEM_DIR}/../SuiteSparse" CACHE PATH
    "Path to the SuiteSparse library.")
set(SuiteSparse_REQUIRED_PACKAGES "BLAS" "METIS"
    CACHE STRING "Additional packages required by SuiteSparse.")

set(ParMETIS_DIR "${MFEM_DIR}/../parmetis-4.0.3" CACHE PATH
    "Path to the ParMETIS library.")
set(ParMETIS_REQUIRED_PACKAGES "METIS" CACHE STRING
    "Additional packages required by ParMETIS.")

set(SuperLUDist_DIR "${MFEM_DIR}/../SuperLU_DIST_5.1.0" CACHE PATH
    "Path to the SuperLU_DIST library.")
# SuperLU_DIST may also depend on "OpenMP", depending on how it was compiled.
set(SuperLUDist_REQUIRED_PACKAGES "MPI" "BLAS" "ParMETIS" CACHE STRING
    "Additional packages required by SuperLU_DIST.")

set(GECKO_DIR "${MFEM_DIR}/../gecko" CACHE PATH "Path to the Gecko library.")

set(GNUTLS_DIR "" CACHE PATH "Path to the GnuTLS library.")

set(NETCDF_DIR "" CACHE PATH "Path to the NetCDF library.")
# May need to add "HDF5" as requirement.
set(NetCDF_REQUIRED_PACKAGES "" CACHE STRING
    "Additional packages required by NetCDF.")

set(PETSC_DIR "${MFEM_DIR}/../petsc" CACHE PATH
    "Path to the PETSc main directory.")
set(PETSC_ARCH "arch-linux2-c-debug" CACHE PATH "PETSc build architecture.")

set(MPFR_DIR "" CACHE PATH "Path to the MPFR library.")

set(CONDUIT_DIR "${MFEM_DIR}/../conduit" CACHE PATH
    "Path to the Conduit library.")
set(Conduit_REQUIRED_PACKAGES "HDF5" CACHE STRING
    "Additional packages required by Conduit.")

set(AXOM_DIR "${MFEM_DIR}/../axom" CACHE PATH "Path to the Axom library.")
# May need to add "Boost" as requirement.
set(Axom_REQUIRED_PACKAGES "Conduit/relay" CACHE STRING
    "Additional packages required by Axom.")

set(BLAS_INCLUDE_DIRS "" CACHE STRING "Path to BLAS headers.")
set(BLAS_LIBRARIES "" CACHE STRING "The BLAS library.")
set(LAPACK_INCLUDE_DIRS "" CACHE STRING "Path to LAPACK headers.")
set(LAPACK_LIBRARIES "" CACHE STRING "The LAPACK library.")

# Some useful variables:
set(CMAKE_SKIP_PREPROCESSED_SOURCE_RULES ON) # Skip *.i rules
set(CMAKE_SKIP_ASSEMBLY_SOURCE_RULES  ON)    # Skip *.s rules
# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefiles.")
