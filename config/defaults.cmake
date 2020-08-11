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

# See the file INSTALL for description of the configuration options.

# Default options. To replace these, copy this file to user.cmake and modify it.

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
endif()

# MFEM options. Set to mimic the default "defaults.mk" file.
option(MFEM_USE_MPI "Enable MPI parallel build" OFF)
option(MFEM_USE_METIS "Enable METIS usage" ${MFEM_USE_MPI})
option(MFEM_USE_EXCEPTIONS "Enable the use of exceptions" OFF)
option(MFEM_USE_ZLIB "Enable zlib for compressed data streams." OFF)
option(MFEM_USE_LIBUNWIND "Enable backtrace for errors." OFF)
option(MFEM_USE_LAPACK "Enable LAPACK usage" OFF)
option(MFEM_THREAD_SAFE "Enable thread safety" OFF)
option(MFEM_USE_OPENMP "Enable the OpenMP backend" OFF)
option(MFEM_USE_LEGACY_OPENMP "Enable legacy OpenMP usage" OFF)
option(MFEM_USE_MEMALLOC "Enable the internal MEMALLOC option." ON)
option(MFEM_USE_SUNDIALS "Enable SUNDIALS usage" OFF)
option(MFEM_USE_MESQUITE "Enable MESQUITE usage" OFF)
option(MFEM_USE_SUITESPARSE "Enable SuiteSparse usage" OFF)
option(MFEM_USE_SUPERLU "Enable SuperLU_DIST usage" OFF)
option(MFEM_USE_STRUMPACK "Enable STRUMPACK usage" OFF)
option(MFEM_USE_GINKGO "Enable Ginkgo usage" OFF)
option(MFEM_USE_GNUTLS "Enable GNUTLS usage" OFF)
option(MFEM_USE_GSLIB "Enable GSLIB usage" OFF)
option(MFEM_USE_NETCDF "Enable NETCDF usage" OFF)
option(MFEM_USE_PETSC "Enable PETSc support." OFF)
option(MFEM_USE_SLEPC "Enable SLEPc support." OFF)
option(MFEM_USE_MPFR "Enable MPFR usage." OFF)
option(MFEM_USE_SIDRE "Enable Axom/Sidre usage" OFF)
option(MFEM_USE_CONDUIT "Enable Conduit usage" OFF)
option(MFEM_USE_PUMI "Enable PUMI" OFF)
option(MFEM_USE_HIOP "Enable HiOp" OFF)
option(MFEM_USE_CUDA "Enable CUDA" OFF)
option(MFEM_USE_OCCA "Enable OCCA" OFF)
option(MFEM_USE_RAJA "Enable RAJA" OFF)
option(MFEM_USE_CEED "Enable CEED" OFF)
option(MFEM_USE_UMPIRE "Enable Umpire" OFF)
option(MFEM_USE_SIMD "Enable use of SIMD intrinsics" OFF)
option(MFEM_USE_ADIOS2 "Enable ADIOS2" OFF)

set(MFEM_MPI_NP 4 CACHE STRING "Number of processes used for MPI tests")

# Allow a user to disable testing, examples, and/or miniapps at CONFIGURE TIME
# if they don't want/need them (e.g. if MFEM is "just a dependency" and all they
# need is the library, building all that stuff adds unnecessary overhead). Note
# that the examples or miniapps can always be built using the targets 'examples'
# or 'miniapps', respectively.
option(MFEM_ENABLE_TESTING "Enable the ctest framework for testing" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" OFF)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)

# Setting CXX/MPICXX on the command line or in user.cmake will overwrite the
# autodetected C++ compiler.
# set(CXX g++)
# set(MPICXX mpicxx)

# Set the target CUDA architecture
set(CUDA_ARCH "sm_60" CACHE STRING "Target CUDA architecture.")

set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The *_DIR paths below will be the first place searched for the corresponding
# headers and library. If these fail, then standard cmake search is performed.
# Note: if the variables are already in the cache, they are not overwritten.

set(HYPRE_DIR "${MFEM_DIR}/../hypre/src/hypre" CACHE PATH
    "Path to the hypre library.")
# If hypre was compiled to depend on BLAS and LAPACK:
# set(HYPRE_REQUIRED_PACKAGES "BLAS" "LAPACK" CACHE STRING
#     "Packages that HYPRE depends on.")

set(METIS_DIR "${MFEM_DIR}/../metis-4.0" CACHE PATH "Path to the METIS library.")

set(LIBUNWIND_DIR "" CACHE PATH "Path to Libunwind.")

# For sundials_nvecparhyp and nvecparallel remember to build with MPI_ENABLED=ON
# and modify cmake variables for hypre for sundials
set(SUNDIALS_DIR "${MFEM_DIR}/../sundials-5.0.0/instdir" CACHE PATH
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

set(STRUMPACK_DIR "${MFEM_DIR}/../STRUMPACK-build" CACHE PATH
    "Path to the STRUMPACK library.")
# STRUMPACK may also depend on "OpenMP", depending on how it was compiled.
# Starting with v2.2.0 of STRUMPACK, ParMETIS and Scotch are optional.
set(STRUMPACK_REQUIRED_PACKAGES "MPI" "MPI_Fortran" "ParMETIS" "METIS"
    "ScaLAPACK" "Scotch/ptscotch/ptscotcherr/scotch/scotcherr" CACHE STRING
    "Additional packages required by STRUMPACK.")
# If the MPI package does not find all required Fortran libraries:
# set(STRUMPACK_REQUIRED_LIBRARIES "gfortran" "mpi_mpifh" CACHE STRING
#     "Additional libraries required by STRUMPACK.")

# The Scotch library, required by STRUMPACK <= v2.1.0, optional in STRUMPACK >=
# v2.2.0.
set(Scotch_DIR "${MFEM_DIR}/../scotch_6.0.4" CACHE PATH
    "Path to the Scotch and PT-Scotch libraries.")
set(Scotch_REQUIRED_PACKAGES "Threads" CACHE STRING
    "Additional packages required by Scotch.")
# Tell the "Threads" package/module to prefer pthreads.
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(Threads_LIB_VARS CMAKE_THREAD_LIBS_INIT)

# The ScaLAPACK library, required by STRUMPACK
set(ScaLAPACK_DIR "${MFEM_DIR}/../scalapack-2.0.2/lib/cmake/scalapack-2.0.2"
    CACHE PATH "Path to the configuration file scalapack-config.cmake")
set(ScaLAPACK_TARGET_NAMES scalapack)
# set(ScaLAPACK_TARGET_FORCE)
# set(ScaLAPACK_IMPORT_CONFIG DEBUG)

set(Ginkgo_DIR "${MFEM_DIR}/../ginkgo" CACHE PATH "Path to the Ginkgo library.")

set(GNUTLS_DIR "" CACHE PATH "Path to the GnuTLS library.")

set(GSLIB_DIR "" CACHE PATH "Path to the GSLIB library.")

set(NETCDF_DIR "" CACHE PATH "Path to the NetCDF library.")
# May need to add "HDF5" as requirement.
set(NetCDF_REQUIRED_PACKAGES "" CACHE STRING
    "Additional packages required by NetCDF.")

set(PETSC_DIR "${MFEM_DIR}/../petsc" CACHE PATH
    "Path to the PETSc main directory.")
set(PETSC_ARCH "arch-linux2-c-debug" CACHE STRING "PETSc build architecture.")

set(SLEPC_DIR "${MFEM_DIR}/../slepc" CACHE PATH
    "Path to the SLEPc main directory.")
set(SLEPC_ARCH "arch-linux2-c-debug" CACHE STRING "SLEPC build architecture.")

set(MPFR_DIR "" CACHE PATH "Path to the MPFR library.")

set(CONDUIT_DIR "${MFEM_DIR}/../conduit" CACHE PATH
    "Path to the Conduit library.")

set(AXOM_DIR "${MFEM_DIR}/../axom" CACHE PATH "Path to the Axom library.")
# May need to add "Boost" as requirement.
set(Axom_REQUIRED_PACKAGES "Conduit/relay/blueprint" CACHE STRING
    "Additional packages required by Axom.")

set(PUMI_DIR "${MFEM_DIR}/../pumi-2.1.0" CACHE STRING
    "Directory where PUMI is installed")

set(HIOP_DIR "${MFEM_DIR}/../hiop/install" CACHE STRING
    "Directory where HiOp is installed")
set(HIOP_REQUIRED_PACKAGES "BLAS" "LAPACK" CACHE STRING
    "Packages that HiOp depends on.")

set(OCCA_DIR "${MFEM_DIR}/../occa" CACHE PATH "Path to OCCA")
set(RAJA_DIR "${MFEM_DIR}/../raja" CACHE PATH "Path to RAJA")
set(CEED_DIR "${MFEM_DIR}/../libCEED" CACHE PATH "Path to libCEED")
set(UMPIRE_DIR "${MFEM_DIR}/../umpire" CACHE PATH "Path to Umpire")

set(BLAS_INCLUDE_DIRS "" CACHE STRING "Path to BLAS headers.")
set(BLAS_LIBRARIES "" CACHE STRING "The BLAS library.")
set(LAPACK_INCLUDE_DIRS "" CACHE STRING "Path to LAPACK headers.")
set(LAPACK_LIBRARIES "" CACHE STRING "The LAPACK library.")

# Some useful variables:
set(CMAKE_SKIP_PREPROCESSED_SOURCE_RULES ON) # Skip *.i rules
set(CMAKE_SKIP_ASSEMBLY_SOURCE_RULES  ON)    # Skip *.s rules
# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefiles.")
