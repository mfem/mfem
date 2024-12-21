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

# This file describes the default MFEM build options.
#
# See the file INSTALL for description of these options.  You can
# customize them below, or copy this file to user.mk and modify it.


# Some choices below are based on the OS type:
NOTMAC := $(subst Darwin,,$(shell uname -s))

ETAGS_BIN = $(shell command -v etags 2> /dev/null)
EGREP_BIN = $(shell command -v egrep 2> /dev/null)

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
#
# If you set MFEM_USE_ENZYME=YES, CUDA_CXX has to be configured to use cuda with
# clang as its host compiler.
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
HIP_FLAGS = --offload-arch=$(HIP_ARCH)
HIP_XCOMPILER =
HIP_XLINKER   = -Wl,

# Flags for generating dependencies.
DEP_FLAGS = -MM -MT

ifneq ($(NOTMAC),)
   AR      = ar
   ARFLAGS = crv
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
   ARFLAGS = Scrv
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
   # Silence unused command line argument warnings when generating dependencies
   # with mpicxx and clang
   DEP_FLAGS := -Wno-unused-command-line-argument $(DEP_FLAGS)
   # Silence "ignoring duplicate libraries" warnings on new (Xcode 15) linker
   ifneq (,$(findstring PROJECT:dyld,$(shell ld -v 2>&1)))
      LDFLAGS_INTERNAL = -Xlinker -no_warn_duplicate_libraries
   endif
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
MFEM_PRECISION         = double
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
MFEM_USE_SUITESPARSE   = NO
MFEM_USE_SUPERLU       = NO
MFEM_USE_SUPERLU5      = NO
MFEM_USE_MUMPS         = NO
MFEM_USE_STRUMPACK     = NO
MFEM_USE_GINKGO        = NO
MFEM_USE_AMGX          = NO
MFEM_USE_MAGMA         = NO
MFEM_USE_GNUTLS        = NO
MFEM_USE_NETCDF        = NO
MFEM_USE_PETSC         = NO
MFEM_USE_SLEPC         = NO
MFEM_USE_MPFR          = NO
MFEM_USE_SIDRE         = NO
MFEM_USE_FMS           = NO
MFEM_USE_CONDUIT       = NO
MFEM_USE_PUMI          = NO
MFEM_USE_HIOP          = NO
MFEM_USE_GSLIB         = NO
MFEM_USE_CUDA          = NO
MFEM_USE_HIP           = NO
MFEM_USE_RAJA          = NO
MFEM_USE_OCCA          = NO
MFEM_USE_CEED          = NO
MFEM_USE_CALIPER       = NO
MFEM_USE_ALGOIM        = NO
MFEM_USE_UMPIRE        = NO
MFEM_USE_SIMD          = NO
MFEM_USE_ADIOS2        = NO
MFEM_USE_MKL_CPARDISO  = NO
MFEM_USE_MKL_PARDISO   = NO
MFEM_USE_MOONOLITH     = NO
MFEM_USE_ADFORWARD     = NO
MFEM_USE_CODIPACK      = NO
MFEM_USE_BENCHMARK     = NO
MFEM_USE_PARELAG       = NO
MFEM_USE_TRIBOL        = NO
MFEM_USE_ENZYME        = NO

# Process MFEM_PRECISION -> MFEM_USE_SINGLE, MFEM_USE_DOUBLE
ifneq ($(filter double Double DOUBLE,$(MFEM_PRECISION)),)
   MFEM_USE_DOUBLE = YES
   MFEM_USE_SINGLE = NO
else ifneq ($(filter single Single SINGLE,$(MFEM_PRECISION)),)
   MFEM_USE_DOUBLE = NO
   MFEM_USE_SINGLE = YES
else ifeq ($(MAKECMDGOALS),config)
   $(error Invalid floating-point precision: \
     MFEM_PRECISION = $(MFEM_PRECISION))
endif

# MPI library compile and link flags
# These settings are used only when building MFEM with MPI + HIP
ifeq ($(MFEM_USE_MPI)$(MFEM_USE_HIP),YESYES)
   # We determine MPI_DIR assuming $(MPICXX) is in $(MPI_DIR)/bin
   MPI_DIR := $(patsubst %/,%,$(dir $(shell which $(MPICXX))))
   MPI_DIR := $(patsubst %/,%,$(dir $(MPI_DIR)))
   MPI_OPT = -I$(MPI_DIR)/include
   MPI_LIB = -L$(MPI_DIR)/lib $(XLINKER)-rpath,$(MPI_DIR)/lib -lmpi
endif

# ROCM/HIP directory such that ROCM/HIP libraries like rocsparse and rocrand are
# found in $(HIP_DIR)/lib, usually as links. Typically, this directory is of
# the form /opt/rocm-X.Y.Z which is called ROCM_PATH by hipconfig.
ifeq ($(MFEM_USE_HIP),YES)
   HIP_DIR := $(patsubst %/,%,$(dir $(shell which $(HIP_CXX))))
   HIP_DIR := $(patsubst %/,%,$(dir $(HIP_DIR)))
   ifeq (,$(wildcard $(HIP_DIR)/lib/librocsparse.*))
      HIP_DIR := $(shell hipconfig --rocmpath 2> /dev/null)
      ifeq (,$(wildcard $(HIP_DIR)/lib/librocsparse.*))
         $(error Unable to determine HIP_DIR. Please set it manually.)
      endif
   endif
endif

# Compile and link options for zlib.
ZLIB_DIR =
ZLIB_OPT = $(if $(ZLIB_DIR),-I$(ZLIB_DIR)/include)
ZLIB_LIB = $(if $(ZLIB_DIR),$(ZLIB_RPATH) -L$(ZLIB_DIR)/lib ,)-lz
ZLIB_RPATH = $(XLINKER)-rpath,$(ZLIB_DIR)/lib

LIBUNWIND_OPT = -g
LIBUNWIND_LIB = $(if $(NOTMAC),-lunwind -ldl,)

# HYPRE library configuration (needed to build the parallel version)
HYPRE_DIR = @MFEM_DIR@/../hypre/src/hypre
HYPRE_OPT = -I$(HYPRE_DIR)/include
HYPRE_LIB = -L$(HYPRE_DIR)/lib -lHYPRE
ifeq (YES,$(MFEM_USE_CUDA))
   # This is only necessary when hypre is built with cuda:
   HYPRE_LIB += -lcusparse -lcurand -lcublas
endif
ifeq (YES,$(MFEM_USE_HIP))
   # This is only necessary when hypre is built with hip:
   HYPRE_LIB += -L$(HIP_DIR)/lib $(XLINKER)-rpath,$(HIP_DIR)/lib\
 -lrocsparse -lrocrand
endif

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
SUNDIALS_DIR = @MFEM_DIR@/../sundials-5.0.0/instdir
# SUNDIALS >= 6.4.0 requires C++14:
ifeq ($(MFEM_USE_SUNDIALS),YES)
   BASE_FLAGS = -std=c++14
endif
SUNDIALS_OPT = -I$(SUNDIALS_DIR)/include
SUNDIALS_LIB = $(XLINKER)-rpath,$(SUNDIALS_DIR)/lib64\
 $(XLINKER)-rpath,$(SUNDIALS_DIR)/lib\
 -L$(SUNDIALS_DIR)/lib64 -L$(SUNDIALS_DIR)/lib\
 -lsundials_arkode -lsundials_cvodes -lsundials_nvecserial -lsundials_kinsol
ifeq ($(MFEM_USE_MPI),YES)
   SUNDIALS_LIB += -lsundials_nvecparallel -lsundials_nvecmpiplusx
endif
ifeq ($(MFEM_USE_CUDA),YES)
   SUNDIALS_LIB += -lsundials_nveccuda
endif
ifeq ($(MFEM_USE_HIP),YES)
   SUNDIALS_LIB += -lsundials_nvechip
endif
SUNDIALS_CORE_PAT = $(subst\
 @MFEM_DIR@,$(MFEM_DIR),$(SUNDIALS_DIR))/lib*/libsundials_core.*
ifeq ($(MFEM_USE_SUNDIALS),YES)
   ifneq ($(wildcard $(SUNDIALS_CORE_PAT)),)
      SUNDIALS_LIB += -lsundials_core
   endif
endif
# If SUNDIALS was built with KLU:
# MFEM_USE_SUITESPARSE = YES

# SuiteSparse library configuration
LIB_RT = $(if $(NOTMAC),-lrt,)
SUITESPARSE_DIR = @MFEM_DIR@/../SuiteSparse
SUITESPARSE_OPT = -I$(SUITESPARSE_DIR)/include
SUITESPARSE_LIB = $(XLINKER)-rpath,$(SUITESPARSE_DIR)/lib\
 -L$(SUITESPARSE_DIR)/lib -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd\
 -lccolamd -lsuitesparseconfig $(LIB_RT) $(METIS_LIB) $(LAPACK_LIB)

# SuperLU library configuration
ifeq ($(MFEM_USE_SUPERLU5),YES)
   SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_5.1.0
   SUPERLU_OPT = -I$(SUPERLU_DIR)/include
   SUPERLU_LIB = $(XLINKER)-rpath,$(SUPERLU_DIR)/lib -L$(SUPERLU_DIR)/lib\
      -lsuperlu_dist_5.1.0
else
   SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_8.1.2
   SUPERLU_OPT = -I$(SUPERLU_DIR)/include
   SUPERLU_LIB = $(XLINKER)-rpath,$(SUPERLU_DIR)/lib64 -L$(SUPERLU_DIR)/lib64\
      -lsuperlu_dist $(LAPACK_LIB)
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
# Additional Fortran library:
# MPI_FORTRAN_LIB += -lgfortran

# MUMPS library configuration
MUMPS_DIR = @MFEM_DIR@/../MUMPS_5.5.0
MUMPS_OPT = -I$(MUMPS_DIR)/include
MUMPS_LIB = $(XLINKER)-rpath,$(MUMPS_DIR)/lib -L$(MUMPS_DIR)/lib
ifeq ($(MFEM_USE_SINGLE),YES)
   MUMPS_LIB += -lsmumps
else
   MUMPS_LIB += -ldmumps
endif
MUMPS_LIB += -lmumps_common -lpord $(SCALAPACK_LIB) $(LAPACK_LIB) $(MPI_FORTRAN_LIB)

# STRUMPACK library configuration
STRUMPACK_DIR = @MFEM_DIR@/../STRUMPACK-build
ifeq ($(MFEM_USE_STRUMPACK),YES)
   BASE_FLAGS = -std=c++14
endif
STRUMPACK_OPT = -I$(STRUMPACK_DIR)/include $(SCOTCH_OPT)
# If STRUMPACK was build with OpenMP support, the following may be need:
# STRUMPACK_OPT += $(OPENMP_OPT)
STRUMPACK_LIB = -L$(STRUMPACK_DIR)/lib -lstrumpack $(MPI_FORTRAN_LIB)\
 $(SCOTCH_LIB) $(SCALAPACK_LIB)

# Ginkgo library configuration
GINKGO_DIR = @MFEM_DIR@/../ginkgo/install
GINKGO_SEARCH_DIR = $(subst @MFEM_DIR@,$(MFEM_DIR),$(GINKGO_DIR))
GINKGO_BUILD_TYPE=Release
ifeq ($(MFEM_USE_GINKGO),YES)
   BASE_FLAGS = -std=c++14
endif
GINKGO_OPT = -isystem $(GINKGO_DIR)/include
GINKGO_LIB_DIR = $(sort $(dir $(wildcard\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.a\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.so\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.dylib\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.dll)))
GINKGO_LINK_LIB_DIR = $(GINKGO_DIR)$(subst $(GINKGO_SEARCH_DIR),,$(GINKGO_LIB_DIR))
ALL_GINKGO_LIBS_DEBUG = $(notdir $(basename $(wildcard\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*d.a\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*d.so\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*d.dylib\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*d.dll)))
ALL_GINKGO_LIBS = $(notdir $(basename $(wildcard\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.a\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.so\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.dylib\
 $(GINKGO_SEARCH_DIR)/lib*/libginkgo*.dll)))
ALL_GINKGO_LIBS_RELEASE = $(filter-out $(ALL_GINKGO_LIBS_DEBUG),$(ALL_GINKGO_LIBS))
GINKGO_LINK = $(subst libginkgo,-lginkgo,$(ALL_GINKGO_LIBS_RELEASE))
ifeq ($(GINKGO_BUILD_TYPE),Debug)
  ifneq (,$(ALL_GINKGO_LIBS_DEBUG))
    GINKGO_LINK = $(subst libginkgo,-lginkgo,$(ALL_GINKGO_LIBS_DEBUG))
  endif
else
endif
GINKGO_LIB = $(XLINKER)-rpath,$(GINKGO_LINK_LIB_DIR) -L$(GINKGO_LINK_LIB_DIR)\
 $(GINKGO_LINK)

# AmgX library configuration
AMGX_DIR = @MFEM_DIR@/../amgx
AMGX_OPT = -I$(AMGX_DIR)/include
AMGX_LIB = -L$(AMGX_DIR)/lib -lamgx -lcusparse -lcusolver -lcublas -lnvToolsExt

# MAGMA library configuration
MAGMA_DIR = @MFEM_DIR@/../magma
MAGMA_OPT = -I$(MAGMA_DIR)/include
MAGMA_LIB = -L$(MAGMA_DIR)/lib -l:libmagma.a -lcublas -lcusparse $(LAPACK_LIB)

# GnuTLS library configuration
GNUTLS_OPT =
GNUTLS_LIB = -lgnutls

# NetCDF library configuration
NETCDF_DIR = $(HOME)/local
HDF5_DIR   = $(HOME)/local
NETCDF_OPT = -I$(NETCDF_DIR)/include -I$(HDF5_DIR)/include $(ZLIB_OPT)
NETCDF_LIB = $(XLINKER)-rpath,$(NETCDF_DIR)/lib -L$(NETCDF_DIR)/lib\
 $(XLINKER)-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib\
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
   PETSC_DEP := $(shell sed -n "s/$(PETSC_LIB_VAR) = *//p" $(PETSC_VARS))
   PETSC_LIB = $(XLINKER)-rpath,$(abspath $(PETSC_DIR))/lib\
      -L$(abspath $(PETSC_DIR))/lib -lpetsc\
      $(subst $(CXX_XLINKER),$(XLINKER),$(PETSC_DEP))
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
   SLEPC_DEP := $(shell sed -n "s/$(SLEPC_LIB_VAR) *= *//p" $(SLEPC_VARS))
   SLEPC_LIB = $(XLINKER)-rpath,$(abspath $(SLEPC_DIR))/$(PETSC_ARCH)/lib\
      -L$(abspath $(SLEPC_DIR))/$(PETSC_ARCH)/lib -lslepc\
      $(subst $(CXX_XLINKER),$(XLINKER),$(SLEPC_DEP))
endif

ifeq ($(MFEM_USE_MOONOLITH),YES)
  include $(MOONOLITH_DIR)/config/moonolith-config.makefile
  MOONOLITH_LIB=$(MOONOLITH_LIBRARIES)
endif

# MPFR library configuration
MPFR_OPT =
MPFR_LIB = -lmpfr

# FMS and required libraries configuration
FMS_DIR = $(MFEM_DIR)/../fms
FMS_OPT = -I$(FMS_DIR)/include
FMS_LIB = -Wl,-rpath,$(FMS_DIR)/lib -L$(FMS_DIR)/lib -lfms

# Conduit and required libraries configuration
CONDUIT_DIR = @MFEM_DIR@/../conduit
CONDUIT_OPT = -I$(CONDUIT_DIR)/include/conduit
CONDUIT_LIB = \
   $(XLINKER)-rpath,$(CONDUIT_DIR)/lib -L$(CONDUIT_DIR)/lib \
   -lconduit -lconduit_relay -lconduit_blueprint  -ldl

# Check if Conduit was built with hdf5 support, by looking
# for the relay hdf5 header
CONDUIT_HDF5_HEADER=$(CONDUIT_DIR)/include/conduit/conduit_relay_hdf5.hpp
ifneq (,$(wildcard $(CONDUIT_HDF5_HEADER)))
   CONDUIT_OPT += -I$(HDF5_DIR)/include
   CONDUIT_LIB += $(XLINKER)-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib \
                  -lhdf5 $(ZLIB_LIB)
endif

# Sidre and required libraries configuration
# Be sure to check the HDF5_DIR (set above) is correct
SIDRE_DIR = @MFEM_DIR@/../axom
SIDRE_OPT = -I$(SIDRE_DIR)/include -I$(CONDUIT_DIR)/include/conduit\
 -I$(HDF5_DIR)/include
SIDRE_LIB = \
   $(XLINKER)-rpath,$(SIDRE_DIR)/lib -L$(SIDRE_DIR)/lib \
   $(XLINKER)-rpath,$(CONDUIT_DIR)/lib -L$(CONDUIT_DIR)/lib \
   $(XLINKER)-rpath,$(HDF5_DIR)/lib -L$(HDF5_DIR)/lib \
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

# CoDiPack
CODIPACK_DIR = @MFEM_DIR@/../CoDiPack
CODIPACK_OPT = -I$(CODIPACK_DIR)
CODIPACK_LIB =

# GSLIB library
GSLIB_DIR = @MFEM_DIR@/../gslib/build
GSLIB_OPT = -I$(GSLIB_DIR)/include
GSLIB_LIB = -L$(GSLIB_DIR)/lib -lgs

# CUDA library configuration
CUDA_OPT =
CUDA_LIB = -lcusparse -lcublas

# HIP library configuration
HIP_OPT =
HIP_LIB = -L$(HIP_DIR)/lib $(XLINKER)-rpath,$(HIP_DIR)/lib -lhipsparse -lhipblas

# OCCA library configuration
OCCA_DIR = @MFEM_DIR@/../occa
OCCA_OPT = -I$(OCCA_DIR)/include
OCCA_LIB = $(XLINKER)-rpath,$(OCCA_DIR)/lib -L$(OCCA_DIR)/lib -locca

# CALIPER library configuration
CALIPER_DIR = @MFEM_DIR@/../caliper
CALIPER_OPT = -I$(CALIPER_DIR)/include
CALIPER_LIB = $(XLINKER)-rpath,$(CALIPER_DIR)/lib64 $(XLINKER)-rpath,$(CALIPER_DIR)/lib -L$(CALIPER_DIR)/lib64 -L$(CALIPER_DIR)/lib -lcaliper

ifdef ADIAK_DIR
   CALIPER_OPT += -I$(ADIAK_DIR)/include
   CALIPER_LIB += $(XLINKER)-rpath,$(ADIAK_DIR)/lib64 $(XLINKER)-rpath,$(ADIAK_DIR)/lib -L$(ADIAK_DIR)/lib64 -L$(ADIAK_DIR)/lib -ladiak
endif
ifdef GOTCHA_DIR
   CALIPER_OPT += -I$(GOTCHA_DIR)/include
   CALIPER_LIB += $(XLINKER)-rpath,$(GOTCHA_DIR)/lib64 $(XLINKER)-rpath,$(GOTCHA_DIR)/lib -L$(GOTCHA_DIR)/lib64 -L$(GOTCHA_DIR)/lib -lgotcha
endif

# BLITZ library configuration
# BLITZ_DIR must be the custom installation folder (-DCMAKE_INSTALL_PREFIX).
BLITZ_DIR = @MFEM_DIR@/../blitz/install
BLITZ_OPT = -I$(BLITZ_DIR)/include
# On intel machines, use /lib64 instead of /lib.
BLITZ_LIB = $(XLINKER)-rpath,$(BLITZ_DIR)/lib -L$(BLITZ_DIR)/lib -lblitz

# ALGOIM library configuration
ALGOIM_DIR = @MFEM_DIR@/../algoim
ALGOIM_OPT = -I$(ALGOIM_DIR)/src $(BLITZ_OPT)
ALGOIM_LIB = $(BLITZ_LIB)

# BENCHMARK library configuration
BENCHMARK_DIR = @MFEM_DIR@/../google-benchmark
BENCHMARK_OPT = -I$(BENCHMARK_DIR)/include
BENCHMARK_LIB = -L$(BENCHMARK_DIR)/lib -lbenchmark -lpthread

# libCEED library configuration
CEED_DIR ?= @MFEM_DIR@/../libCEED
CEED_OPT = -I$(CEED_DIR)/include
CEED_LIB = $(XLINKER)-rpath,$(CEED_DIR)/lib -L$(CEED_DIR)/lib -lceed

# RAJA library configuration
ifeq ($(MFEM_USE_RAJA),YES)
   BASE_FLAGS = -std=c++14
endif
RAJA_DIR = @MFEM_DIR@/../raja
RAJA_OPT = -I$(RAJA_DIR)/include
ifdef CUB_DIR
   RAJA_OPT += -I$(CUB_DIR)
endif

CAMP_LIB = -lcamp
ifdef CAMP_DIR
   RAJA_OPT += -I$(CAMP_DIR)/include
   CAMP_LIB = $(XLINKER)-rpath,$(CAMP_DIR)/lib -L$(CAMP_DIR)/lib -lcamp
endif
RAJA_LIB = $(XLINKER)-rpath,$(RAJA_DIR)/lib -L$(RAJA_DIR)/lib -lRAJA $(CAMP_LIB)

# UMPIRE library configuration
ifeq ($(MFEM_USE_UMPIRE),YES)
   BASE_FLAGS = -std=c++14
endif
UMPIRE_DIR = @MFEM_DIR@/../umpire
UMPIRE_OPT = -I$(UMPIRE_DIR)/include $(if $(CAMP_DIR), -I$(CAMP_DIR)/include)
UMPIRE_LIB = -L$(UMPIRE_DIR)/lib -lumpire $(CAMP_LIB)

# MKL CPardiso library configuration
MKL_CPARDISO_DIR ?=
MKL_MPI_WRAPPER ?= mkl_blacs_mpich_lp64
MKL_LIBRARY_SUBDIR ?= lib
MKL_CPARDISO_OPT = -I$(MKL_CPARDISO_DIR)/include
MKL_CPARDISO_LIB = $(XLINKER)-rpath,$(MKL_CPARDISO_DIR)/$(MKL_LIBRARY_SUBDIR)\
   -L$(MKL_CPARDISO_DIR)/$(MKL_LIBRARY_SUBDIR) -l$(MKL_MPI_WRAPPER)\
   -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# MKL Pardiso library configuration
MKL_PARDISO_DIR ?=
MKL_LIBRARY_SUBDIR ?= lib
MKL_PARDISO_OPT = -I$(MKL_PARDISO_DIR)/include
MKL_PARDISO_LIB = $(XLINKER)-rpath,$(MKL_PARDISO_DIR)/$(MKL_LIBRARY_SUBDIR)\
   -L$(MKL_PARDISO_DIR)/$(MKL_LIBRARY_SUBDIR)\
   -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

# PARELAG library configuration
PARELAG_DIR = @MFEM_DIR@/../parelag
PARELAG_OPT = -I$(PARELAG_DIR)/src -I$(PARELAG_DIR)/build/src
PARELAG_LIB = -L$(PARELAG_DIR)/build/src -lParELAG

# Tribol library configuration
ifeq ($(MFEM_USE_TRIBOL),YES)
   BASE_FLAGS = -std=c++14
endif
AXOM_DIR = @MFEM_DIR@/../axom
TRIBOL_DIR = @MFEM_DIR@/../tribol
TRIBOL_OPT = -I$(TRIBOL_DIR)/include -I$(AXOM_DIR)/include
TRIBOL_LIB = -L$(TRIBOL_DIR)/lib -ltribol -lredecomp -L$(AXOM_DIR)/lib -laxom_mint\
   -laxom_slam -laxom_slic -laxom_core

# Enzyme configuration

# If you want to enable automatic differentiation at compile time, use the
# options below, adapted to your configuration. To be more flexible, we
# recommend using the Enzyme plugin during link time optimization. One option is
# to add your options to the global compiler/linker flags like
#
# BASE_FLAGS += -flto
# CXX_XLINKER += -fuse-ld=lld -Wl,--lto-legacy-pass-manager\
#                -Wl,-mllvm=-load=$(ENZYME_DIR)/LLDEnzyme-$(ENZYME_VERSION).so -Wl,
#
ENZYME_DIR ?= @MFEM_DIR@/../enzyme
ENZYME_VERSION ?= 14
ENZYME_OPT = -fno-experimental-new-pass-manager -Xclang -load -Xclang $(ENZYME_DIR)/ClangEnzyme-$(ENZYME_VERSION).so
ENZYME_LIB = ""

# If YES, enable some informational messages
VERBOSE = NO

# Optional build tag
MFEM_BUILD_TAG = $(shell uname -snm)
