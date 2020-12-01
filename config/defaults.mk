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

# This file describes the default MFEM build options.
#
# See the file INSTALL for description of these options.  You can
# customize them below, or copy this file to user.mk and modify it.


# Some choices below are based on the OS type:
NOTMAC := $(subst Darwin,,$(shell uname -s))

CXX = g++
MPICXX = mpicxx

BASE_FLAGS  = -std=c++11
OPTIM_FLAGS = -O3 $(BASE_FLAGS)
DEBUG_FLAGS = -g $(XCOMPILER)-Wall $(BASE_FLAGS)

# Prefixes for passing flags to the compiler and linker when using CXX or MPICXX
CXX_XCOMPILER =
CXX_XLINKER   = -Wl,

# Destination location of make install
# PREFIX = $(HOME)/mfem
PREFIX = ./mfem
# Install program
INSTALL = /usr/bin/install

STATIC = YES
SHARED = NO

# CUDA configuration options
CUDA_CXX = nvcc
CUDA_ARCH = sm_60
CUDA_FLAGS = -x=cu --expt-extended-lambda -arch=$(CUDA_ARCH)
# Prefixes for passing flags to the host compiler and linker when using CUDA_CXX
CUDA_XCOMPILER = -Xcompiler=
CUDA_XLINKER   = -Xlinker=

# HIP configuration options
HIP_CXX = hipcc
# The HIP_ARCH option specifies the AMD GPU processor, similar to CUDA_ARCH. For
# example: gfx600 (tahiti), gfx700 (kaveri), gfx701 (hawaii), gfx801 (carrizo),
# gfx900, gfx1010, etc.
HIP_ARCH = gfx900
HIP_FLAGS = --amdgpu-target=$(HIP_ARCH)
HIP_XCOMPILER =
HIP_XLINKER   = -Wl,

ifneq ($(NOTMAC),)
   AR      = ar
   ARFLAGS = cruv
   RANLIB  = ranlib
   PICFLAG = $(XCOMPILER)-fPIC
   SO_EXT  = so
   SO_VER  = so.$(MFEM_VERSION_STRING)
   BUILD_SOFLAGS = -shared $(XLINKER)-soname,libmfem.$(SO_VER)
   BUILD_RPATH = $(XLINKER)-rpath,$(BUILD_REAL_DIR)
   INSTALL_SOFLAGS = $(BUILD_SOFLAGS)
   INSTALL_RPATH = $(XLINKER)-rpath,@MFEM_LIB_DIR@
else
   # Silence "has no symbols" warnings on Mac OS X
   AR      = ar
   ARFLAGS = Scruv
   RANLIB  = ranlib -no_warning_for_no_symbols
   PICFLAG = $(XCOMPILER)-fPIC
   SO_EXT  = dylib
   SO_VER  = $(MFEM_VERSION_STRING).dylib
   MAKE_SOFLAGS = $(XLINKER)-dylib,-install_name,$(1)/libmfem.$(SO_VER),\
      -compatibility_version,$(MFEM_VERSION_STRING),\
      -current_version,$(MFEM_VERSION_STRING),\
      -undefined,dynamic_lookup
   BUILD_SOFLAGS = $(subst $1 ,,$(call MAKE_SOFLAGS,$(BUILD_REAL_DIR)))
   BUILD_RPATH = $(XLINKER)-undefined,dynamic_lookup
   INSTALL_SOFLAGS = $(subst $1 ,,$(call MAKE_SOFLAGS,$(MFEM_LIB_DIR)))
   INSTALL_RPATH = $(XLINKER)-undefined,dynamic_lookup
endif

# Set CXXFLAGS to overwrite the default selection of DEBUG_FLAGS/OPTIM_FLAGS
# CXXFLAGS = -O3 -march=native

# Optional extra compile flags, in addition to CXXFLAGS:
# CPPFLAGS =

# Library configurations:
# Note: symbols of the form @VAR@ will be replaced by $(VAR) in derived
#       variables, like MFEM_FLAGS, defined in config.mk.

# Command used to launch MPI jobs
MFEM_MPIEXEC    = mpirun
MFEM_MPIEXEC_NP = -np
# Number of mpi tasks for parallel jobs
MFEM_MPI_NP = 4

# MFEM configuration options: YES/NO values, which are exported to config.mk and
# config.hpp. The values below are the defaults for generating the actual values
# in config.mk and config.hpp.

MFEM_USE_MPI           = NO
MFEM_USE_METIS         = $(MFEM_USE_MPI)
MFEM_USE_METIS_5       = NO
MFEM_DEBUG             = NO
MFEM_USE_EXCEPTIONS    = NO
MFEM_USE_ZLIB          = NO
MFEM_USE_LIBUNWIND     = NO
MFEM_USE_LAPACK        = NO
MFEM_THREAD_SAFE       = NO
MFEM_USE_OPENMP        = NO
MFEM_USE_LEGACY_OPENMP = NO
MFEM_USE_MEMALLOC      = YES
MFEM_TIMER_TYPE        = $(if $(NOTMAC),2,4)
MFEM_USE_SUNDIALS      = NO
MFEM_USE_MESQUITE      = NO
MFEM_USE_SUITESPARSE   = NO
MFEM_USE_SUPERLU       = NO
MFEM_USE_SUPERLU5      = NO
MFEM_USE_MUMPS         = NO
MFEM_USE_STRUMPACK     = NO
MFEM_USE_GINKGO        = NO
MFEM_USE_AMGX          = NO
MFEM_USE_GNUTLS        = NO
MFEM_USE_NETCDF        = NO
MFEM_USE_PETSC         = NO
MFEM_USE_SLEPC         = NO
MFEM_USE_MPFR          = NO
MFEM_USE_SIDRE         = NO
MFEM_USE_CONDUIT       = NO
MFEM_USE_PUMI          = NO
MFEM_USE_HIOP          = NO
MFEM_USE_GSLIB         = NO
MFEM_USE_CUDA          = NO
MFEM_USE_HIP           = NO
MFEM_USE_RAJA          = NO
MFEM_USE_OCCA          = NO
MFEM_USE_CEED          = NO
MFEM_USE_UMPIRE        = NO
MFEM_USE_SIMD          = NO
MFEM_USE_ADIOS2        = NO
MFEM_USE_MKL_CPARDISO  = NO

# MPI library compile and link flags
# These settings are used only when building MFEM with MPI + HIP
ifeq ($(MFEM_USE_MPI)$(MFEM_USE_HIP),YESYES)
   # We determine MPI_DIR assuming $(MPICXX) is in $(MPI_DIR)/bin
   MPI_DIR := $(patsubst %/,%,$(dir $(shell which $(MPICXX))))
   MPI_DIR := $(patsubst %/,%,$(dir $(MPI_DIR)))
   MPI_OPT = -I$(MPI_DIR)/include
   MPI_LIB = -L$(MPI_DIR)/lib $(XLINKER)-rpath,$(MPI_DIR)/lib -lmpi
endif

# Compile and link options for zlib.
ZLIB_DIR =
ZLIB_OPT = $(if $(ZLIB_DIR),-I$(ZLIB_DIR)/include)
ZLIB_LIB = $(if $(ZLIB_DIR),$(ZLIB_RPATH) -L$(ZLIB_DIR)/lib ,)-lz
ZLIB_RPATH = -Wl,-rpath,$(ZLIB_DIR)/lib

LIBUNWIND_OPT = -g
LIBUNWIND_LIB = $(if $(NOTMAC),-lunwind -ldl,)

# HYPRE library configuration (needed to build the parallel version)
HYPRE_DIR = @MFEM_DIR@/../hypre/src/hypre
HYPRE_OPT = -I$(HYPRE_DIR)/include
HYPRE_LIB = -L$(HYPRE_DIR)/lib -lHYPRE

# METIS library configuration
ifeq ($(MFEM_USE_SUPERLU)$(MFEM_USE_STRUMPACK)$(MFEM_USE_MUMPS),NONONO)
   ifeq ($(MFEM_USE_METIS_5),NO)
     METIS_DIR = @MFEM_DIR@/../metis-4.0
     METIS_OPT =
     METIS_LIB = -L$(METIS_DIR) -lmetis
   else
     METIS_DIR = @MFEM_DIR@/../metis-5.0
     METIS_OPT = -I$(METIS_DIR)/include
     METIS_LIB = -L$(METIS_DIR)/lib -lmetis
   endif
else
   # ParMETIS: currently needed by SuperLU or STRUMPACK. We assume that METIS 5
   # (included with ParMETIS) is installed in the same location.
   # Starting with STRUMPACK v2.2.0, ParMETIS is an optional dependency while
   # METIS is still required.
   METIS_DIR = @MFEM_DIR@/../parmetis-4.0.3
   METIS_OPT = -I$(METIS_DIR)/include
   METIS_LIB = -L$(METIS_DIR)/lib -lparmetis -lmetis
   MFEM_USE_METIS_5 = YES
endif

# LAPACK library configuration
LAPACK_OPT =
LAPACK_LIB = $(if $(NOTMAC),-llapack -lblas,-framework Accelerate)

# OpenMP configuration
OPENMP_OPT = $(XCOMPILER)-fopenmp
OPENMP_LIB =

# Used when MFEM_TIMER_TYPE = 2
POSIX_CLOCKS_LIB = -lrt

# SUNDIALS library configuration
# For sundials_nvecmpiplusx and nvecparallel remember to build with MPI_ENABLE=ON
# and modify cmake variables for hypre for sundials
SUNDIALS_DIR    = @MFEM_DIR@/../sundials-5.0.0/instdir
SUNDIALS_OPT    = -I$(SUNDIALS_DIR)/include
SUNDIALS_LIBDIR = $(wildcard $(SUNDIALS_DIR)/lib*)
SUNDIALS_LIB    = $(XLINKER)-rpath,$(SUNDIALS_LIBDIR) -L$(SUNDIALS_LIBDIR)\
 -lsundials_arkode -lsundials_cvodes -lsundials_nvecserial -lsundials_kinsol

ifeq ($(MFEM_USE_MPI),YES)
   SUNDIALS_LIB += -lsundials_nvecparallel -lsundials_nvecmpiplusx
endif
ifeq ($(MFEM_USE_CUDA),YES)
   SUNDIALS_LIB += -lsundials_nveccuda
endif
# If SUNDIALS was built with KLU:
# MFEM_USE_SUITESPARSE = YES

# MESQUITE library configuration
MESQUITE_DIR = @MFEM_DIR@/../mesquite-2.99
MESQUITE_OPT = -I$(MESQUITE_DIR)/include
MESQUITE_LIB = -L$(MESQUITE_DIR)/lib -lmesquite

# SuiteSparse library configuration
LIB_RT = $(if $(NOTMAC),-lrt,)
SUITESPARSE_DIR = @MFEM_DIR@/../SuiteSparse
SUITESPARSE_OPT = -I$(SUITESPARSE_DIR)/include
SUITESPARSE_LIB = -Wl,-rpath,$(SUITESPARSE_DIR)/lib -L$(SUITESPARSE_DIR)/lib\
 -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd\
 -lsuitesparseconfig $(LIB_RT) $(METIS_LIB) $(LAPACK_LIB)

# SuperLU library configuration
ifeq ($(MFEM_USE_SUPERLU5),YES)
   SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_5.1.0
   SUPERLU_OPT = -I$(SUPERLU_DIR)/include
   SUPERLU_LIB = -Wl,-rpath,$(SUPERLU_DIR)/lib -L$(SUPERLU_DIR)/lib -lsuperlu_dist_5.1.0
else
   SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_6.3.1
   SUPERLU_OPT = -I$(SUPERLU_DIR)/include
   SUPERLU_LIB = -Wl,-rpath,$(SUPERLU_DIR)/lib64 -L$(SUPERLU_DIR)/lib64 -lsuperlu_dist -lblas
endif

# SCOTCH library configuration (required by STRUMPACK <= v2.1.0, optional in
# STRUMPACK >= v2.2.0)
SCOTCH_DIR = @MFEM_DIR@/../scotch_6.0.4
SCOTCH_OPT = -I$(SCOTCH_DIR)/include
SCOTCH_LIB = -L$(SCOTCH_DIR)/lib -lptscotch -lptscotcherr -lscotch -lscotcherr\
 -lpthread

# SCALAPACK library configuration (required by STRUMPACK and MUMPS)
SCALAPACK_DIR = @MFEM_DIR@/../scalapack-2.0.2
SCALAPACK_OPT = -I$(SCALAPACK_DIR)/SRC
SCALAPACK_LIB = -L$(SCALAPACK_DIR)/lib -lscalapack $(LAPACK_LIB)

# MPI Fortran library, needed e.g. by STRUMPACK or MUMPS
# MPICH:
MPI_FORTRAN_LIB = -lmpifort
# OpenMPI:
# MPI_FORTRAN_LIB = -lmpi_mpifh
# Additional Fortan library:
# MPI_FORTRAN_LIB += -lgfortran

# MUMPS library configuration
MUMPS_DIR = @MFEM_DIR@/../MUMPS_5.2.0
MUMPS_OPT = -I$(MUMPS_DIR)/include
MUMPS_LIB = -Wl,-rpath,$(MUMPS_DIR)/lib -L$(MUMPS_DIR)/lib -ldmumps\
 -lmumps_common -lpord $(SCALAPACK_LIB) $(LAPACK_LIB) $(MPI_FORTRAN_LIB)

# STRUMPACK library configuration
STRUMPACK_DIR = @MFEM_DIR@/../STRUMPACK-build
STRUMPACK_OPT = -I$(STRUMPACK_DIR)/include $(SCOTCH_OPT)
# If STRUMPACK was build with OpenMP support, the following may be need:
# STRUMPACK_OPT += $(OPENMP_OPT)
STRUMPACK_LIB = -L$(STRUMPACK_DIR)/lib -lstrumpack $(MPI_FORTRAN_LIB)\
 $(SCOTCH_LIB) $(SCALAPACK_LIB)

# Ginkgo library configuration (currently not needed)
GINKGO_DIR = @MFEM_DIR@/../ginkgo/install
GINKGO_OPT = -isystem $(GINKGO_DIR)/include
GINKGO_LIB = $(XLINKER)-rpath,$(GINKGO_DIR)/lib -L$(GINKGO_DIR)/lib -lginkgo\
 -lginkgo_omp -lginkgo_cuda -lginkgo_reference

# AmgX library configuration
AMGX_DIR = @MFEM_DIR@/../amgx
AMGX_OPT = -I$(AMGX_DIR)/include
AMGX_LIB = -lcusparse -lcusolver -lcublas -lnvToolsExt -L$(AMGX_DIR)/lib -lamgx

# GnuTLS library configuration
GNUTLS_OPT =
GNUTLS_LIB = -lgnutls

# NetCDF library configuration
NETCDF_DIR = $(HOME)/local
HDF5_DIR   = $(HOME)/local
NETCDF_OPT = -I$(NETCDF_DIR)/include -I$(HDF5_DIR)/include $(ZLIB_OPT)
NETCDF_LIB = -Wl,-rpath,$(NETCDF_DIR)/lib -L$(NETCDF_DIR)/lib\
 -Wl,-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib\
 -lnetcdf -lhdf5_hl -lhdf5 $(ZLIB_LIB)

# PETSc library configuration (version greater or equal to 3.8 or the dev branch)
PETSC_ARCH := arch-linux2-c-debug
PETSC_DIR  := $(MFEM_DIR)/../petsc/$(PETSC_ARCH)
PETSC_VARS := $(PETSC_DIR)/lib/petsc/conf/petscvariables
PETSC_FOUND := $(if $(wildcard $(PETSC_VARS)),YES,)
PETSC_INC_VAR = PETSC_CC_INCLUDES
PETSC_LIB_VAR = PETSC_EXTERNAL_LIB_BASIC
ifeq ($(PETSC_FOUND),YES)
   PETSC_OPT := $(shell sed -n "s/$(PETSC_INC_VAR) = *//p" $(PETSC_VARS))
   PETSC_LIB := $(shell sed -n "s/$(PETSC_LIB_VAR) = *//p" $(PETSC_VARS))
   PETSC_LIB := -Wl,-rpath,$(abspath $(PETSC_DIR))/lib\
      -L$(abspath $(PETSC_DIR))/lib -lpetsc $(PETSC_LIB)
endif

SLEPC_DIR := $(MFEM_DIR)/../slepc
SLEPC_VARS := $(SLEPC_DIR)/lib/slepc/conf/slepc_variables
SLEPC_FOUND := $(if $(wildcard $(SLEPC_VARS)),YES,)
SLEPC_INC_VAR = SLEPC_INCLUDE
SLEPC_LIB_VAR = SLEPC_EXTERNAL_LIB
ifeq ($(SLEPC_FOUND),YES)
   SLEPC_OPT := $(shell sed -n "s/$(SLEPC_INC_VAR) *= *//p" $(SLEPC_VARS))
   # Some additional external libraries might be defined in this file
   -include ${SLEPC_DIR}/${PETSC_ARCH}/lib/slepc/conf/slepcvariables
   SLEPC_LIB := $(shell sed -n "s/$(SLEPC_LIB_VAR) *= *//p" $(SLEPC_VARS))
   SLEPC_LIB := -Wl,-rpath,$(abspath $(SLEPC_DIR))/$(PETSC_ARCH)/lib\
      -L$(abspath $(SLEPC_DIR))/$(PETSC_ARCH)/lib -lslepc $(SLEPC_LIB)
endif

# MPFR library configuration
MPFR_OPT =
MPFR_LIB = -lmpfr

# Conduit and required libraries configuration
CONDUIT_DIR = @MFEM_DIR@/../conduit
CONDUIT_OPT = -I$(CONDUIT_DIR)/include/conduit
CONDUIT_LIB = \
   -Wl,-rpath,$(CONDUIT_DIR)/lib -L$(CONDUIT_DIR)/lib \
   -lconduit -lconduit_relay -lconduit_blueprint  -ldl

# Check if Conduit was built with hdf5 support, by looking
# for the relay hdf5 header
CONDUIT_HDF5_HEADER=$(CONDUIT_DIR)/include/conduit/conduit_relay_hdf5.hpp
ifneq (,$(wildcard $(CONDUIT_HDF5_HEADER)))
   CONDUIT_OPT += -I$(HDF5_DIR)/include
   CONDUIT_LIB += -Wl,-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib \
                  -lhdf5 $(ZLIB_LIB)
endif

# Sidre and required libraries configuration
# Be sure to check the HDF5_DIR (set above) is correct
SIDRE_DIR = @MFEM_DIR@/../axom
SIDRE_OPT = -I$(SIDRE_DIR)/include -I$(CONDUIT_DIR)/include/conduit\
 -I$(HDF5_DIR)/include
SIDRE_LIB = \
   -Wl,-rpath,$(SIDRE_DIR)/lib -L$(SIDRE_DIR)/lib \
   -Wl,-rpath,$(CONDUIT_DIR)/lib -L$(CONDUIT_DIR)/lib \
   -Wl,-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib \
   -laxom -lconduit -lconduit_relay -lconduit_blueprint -lhdf5 $(ZLIB_LIB) -ldl

# PUMI
# Note that PUMI_DIR is needed -- it is used to check for gmi_sim.h
PUMI_DIR = @MFEM_DIR@/../pumi-2.1.0
PUMI_OPT = -I$(PUMI_DIR)/include
PUMI_LIB = -L$(PUMI_DIR)/lib -lpumi -lcrv -lma -lmds -lapf -lpcu -lgmi -lparma\
   -llion -lmth -lapf_zoltan -lspr

# HIOP
HIOP_DIR = @MFEM_DIR@/../hiop/install
HIOP_OPT = -I$(HIOP_DIR)/include
HIOP_LIB = -L$(HIOP_DIR)/lib -lhiop $(LAPACK_LIB)

# GSLIB library
GSLIB_DIR = @MFEM_DIR@/../gslib/build
GSLIB_OPT = -I$(GSLIB_DIR)/include
GSLIB_LIB = -L$(GSLIB_DIR)/lib -lgs

# CUDA library configuration
CUDA_OPT =
CUDA_LIB = -lcusparse

# HIP library configuration (currently not needed)
HIP_OPT =
HIP_LIB =

# OCCA library configuration
OCCA_DIR = @MFEM_DIR@/../occa
OCCA_OPT = -I$(OCCA_DIR)/include
OCCA_LIB = $(XLINKER)-rpath,$(OCCA_DIR)/lib -L$(OCCA_DIR)/lib -locca

# libCEED library configuration
CEED_DIR ?= @MFEM_DIR@/../libCEED
CEED_OPT = -I$(CEED_DIR)/include
CEED_LIB = $(XLINKER)-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

# RAJA library configuration
RAJA_DIR = @MFEM_DIR@/../raja
RAJA_OPT = -I$(RAJA_DIR)/include
ifdef CUB_DIR
   RAJA_OPT += -I$(CUB_DIR)
endif
RAJA_LIB = $(XLINKER)-rpath,$(RAJA_DIR)/lib -L$(RAJA_DIR)/lib -lRAJA

# UMPIRE library configuration
UMPIRE_DIR = @MFEM_DIR@/../umpire
UMPIRE_OPT = -I$(UMPIRE_DIR)/include
UMPIRE_LIB = -L$(UMPIRE_DIR)/lib -lumpire

# MKL CPardiso library configuration
MKL_CPARDISO_DIR ?=
MKL_MPI_WRAPPER ?= mkl_blacs_mpich_lp64
MKL_LIBRARY_SUBDIR ?= lib
MKL_CPARDISO_OPT = -I$(MKL_CPARDISO_DIR)/include
MKL_CPARDISO_LIB = -Wl,-rpath,$(MKL_CPARDISO_DIR)/$(MKL_LIBRARY_SUBDIR)\
                   -L$(MKL_CPARDISO_DIR)/$(MKL_LIBRARY_SUBDIR) -l$(MKL_MPI_WRAPPER)\
                   -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# If YES, enable some informational messages
VERBOSE = NO

# Optional build tag
MFEM_BUILD_TAG = $(shell uname -snm)
