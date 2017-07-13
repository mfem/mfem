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

# This file describes the default MFEM build options.
#
# See the file INSTALL for description of these options.  You can
# customize them below, or copy this file to user.mk and modify it.


# Some choices below are based on the OS type:
NOTMAC := $(subst Darwin,,$(shell uname -s))

CXX = g++
MPICXX = mpicxx

OPTIM_FLAGS = -O3
DEBUG_FLAGS = -g -Wall

# Destination location of make install
# PREFIX = $(HOME)/mfem
PREFIX = ./mfem
# Install program
INSTALL = /usr/bin/install

ifneq ($(NOTMAC),)
   AR      = ar
   ARFLAGS = cruv
   RANLIB  = ranlib
else
   # Silence "has no symbols" warnings on Mac OS X
   AR      = ar
   ARFLAGS = Scruv
   RANLIB  = ranlib -no_warning_for_no_symbols
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

# MFEM configuration options: YES/NO values, which are exported to config.mk and
# config.hpp. The values below are the defaults for generating the actual values
# in config.mk and config.hpp.

MFEM_USE_MPI         = NO
MFEM_USE_METIS_5     = NO
MFEM_DEBUG           = NO
MFEM_USE_GZSTREAM    = NO
MFEM_USE_LIBUNWIND   = NO
MFEM_USE_LAPACK      = NO
MFEM_THREAD_SAFE     = NO
MFEM_USE_OPENMP      = NO
MFEM_USE_MEMALLOC    = YES
MFEM_TIMER_TYPE      = $(if $(NOTMAC),2,4)
MFEM_USE_SUNDIALS    = NO
MFEM_USE_MESQUITE    = NO
MFEM_USE_SUITESPARSE = NO
MFEM_USE_SUPERLU     = NO
MFEM_USE_STRUMPACK   = NO
MFEM_USE_GECKO       = NO
MFEM_USE_GNUTLS      = NO
MFEM_USE_NETCDF      = NO
MFEM_USE_PETSC       = NO
MFEM_USE_MPFR        = NO
MFEM_USE_SIDRE       = NO
MFEM_USE_OCCA        = NO
MFEM_USE_ACROTENSOR  = NO

LIBUNWIND_OPT = -g
LIBUNWIND_LIB = $(if $(NOTMAC),-lunwind -ldl,)

# HYPRE library configuration (needed to build the parallel version)
HYPRE_DIR = @MFEM_DIR@/../hypre-2.10.0b/src/hypre
HYPRE_OPT = -I$(HYPRE_DIR)/include
HYPRE_LIB = -L$(HYPRE_DIR)/lib -lHYPRE

# METIS library configuration
ifeq ($(MFEM_USE_SUPERLU)$(MFEM_USE_STRUMPACK),NONO)
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
   # ParMETIS currently needed only with SuperLU. We assume that METIS 5
   # (included with ParMETIS) is installed in the same location.
   METIS_DIR = @MFEM_DIR@/../parmetis-4.0.3
   METIS_OPT = -I$(METIS_DIR)/include
   METIS_LIB = -L$(METIS_DIR)/lib -lparmetis -lmetis
   MFEM_USE_METIS_5 = YES
endif

# LAPACK library configuration
LAPACK_OPT =
LAPACK_LIB = $(if $(NOTMAC),-llapack -lblas,-framework Accelerate)

# OpenMP configuration
OPENMP_OPT = -fopenmp
OPENMP_LIB =

# Used when MFEM_TIMER_TYPE = 2
POSIX_CLOCKS_LIB = -lrt

# SUNDIALS library configuration
SUNDIALS_DIR = @MFEM_DIR@/../sundials-2.7.0
SUNDIALS_OPT = -I$(SUNDIALS_DIR)/include
SUNDIALS_LIB = -Wl,-rpath,$(SUNDIALS_DIR)/lib -L$(SUNDIALS_DIR)/lib\
  -lsundials_arkode -lsundials_cvode -lsundials_nvecserial -lsundials_kinsol

ifeq ($(MFEM_USE_MPI),YES)
   SUNDIALS_LIB += -lsundials_nvecparhyp -lsundials_nvecparallel
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
SUITESPARSE_LIB = -L$(SUITESPARSE_DIR)/lib -lklu -lbtf -lumfpack -lcholmod\
 -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig $(LIB_RT) $(METIS_LIB)\
 $(LAPACK_LIB)

# SuperLU library configuration
SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_5.1.0
SUPERLU_OPT = -I$(SUPERLU_DIR)/SRC
SUPERLU_LIB = -L$(SUPERLU_DIR)/SRC -lsuperlu_dist

# SCOTCH library configuration
SCOTCH_DIR = @MFEM_DIR@/../scotch_6.0.4
SCOTCH_OPT = -I$(SCOTCH_DIR)/include
SCOTCH_LIB = -L$(SCOTCH_DIR)/lib -lptscotch -lptscotcherr -lptscotcherrexit\
 -lptscotchparmetis -lscotch -lscotcherr -lscotcherrexit -lscotchmetis

# SCALAPACK library configuration
SCALAPACK_DIR = @MFEM_DIR@/../scalapack_2.0.2
SCALAPACK_OPT = -I$(SCALAPACK_DIR)/SRC
SCALAPACK_LIB = -L$(SCALAPACK_DIR) -lscalapack

# STRUMPACK library configuration
STRUMPACK_DIR = @MFEM_DIR@/../STRUMPACK-build
STRUMPACK_OPT = -I$(STRUMPACK_DIR)/include
STRUMPACK_LIB = -L$(STRUMPACK_DIR)/lib -lstrumpack $(SCOTCH_LIB)\
 $(SCALAPACK_LIB)

# Gecko library configuration
GECKO_DIR = @MFEM_DIR@/../gecko
GECKO_OPT = -I$(GECKO_DIR)/inc
GECKO_LIB = -L$(GECKO_DIR)/lib -lgecko

# GnuTLS library configuration
GNUTLS_OPT =
GNUTLS_LIB = -lgnutls

# NetCDF library configuration
NETCDF_DIR  = $(HOME)/local
HDF5_DIR    = $(HOME)/local
ZLIB_DIR    = $(HOME)/local
NETCDF_OPT  = -I$(NETCDF_DIR)/include
NETCDF_LIB  = -L$(NETCDF_DIR)/lib -lnetcdf -L$(HDF5_DIR)/lib -lhdf5_hl -lhdf5\
 -L$(ZLIB_DIR)/lib -lz

# PETSc library configuration (version greater or equal to 3.8 or the dev branch)
ifeq ($(MFEM_USE_PETSC),YES)
   PETSC_DIR := $(MFEM_DIR)/../petsc/arch-linux2-c-debug
   PETSC_PC  := $(PETSC_DIR)/lib/pkgconfig/PETSc.pc
   $(if $(wildcard $(PETSC_PC)),,$(error PETSc config not found - $(PETSC_PC)))
   PETSC_OPT := $(shell sed -n "s/Cflags: *//p" $(PETSC_PC))
   PETSC_LIBS_PRIVATE := $(shell sed -n "s/Libs\.private: *//p" $(PETSC_PC))
   PETSC_LIB := -Wl,-rpath -Wl,$(abspath $(PETSC_DIR))/lib\
 -L$(abspath $(PETSC_DIR))/lib -lpetsc $(PETSC_LIBS_PRIVATE)
endif

# MPFR library configuration
MPFR_OPT =
MPFR_LIB = -lmpfr

# Sidre and required libraries configuration
# Be sure to check the HDF5_DIR (set above) is correct
SIDRE_DIR = @MFEM_DIR@/../axom
CONDUIT_DIR = @MFEM_DIR@/../conduit
SIDRE_OPT = -I$(SIDRE_DIR)/include -I$(CONDUIT_DIR)/include/conduit\
 -I$(HDF5_DIR)/include
SIDRE_LIB = \
   -L$(SIDRE_DIR)/lib \
   -L$(CONDUIT_DIR)/lib \
   -Wl,-rpath -Wl,$(CONDUIT_DIR)/lib \
   -L$(HDF5_DIR)/lib \
   -Wl,-rpath -Wl,$(HDF5_DIR)/lib \
   -lsidre -lslic -laxom_utils -lconduit -lconduit_relay -lhdf5 -lz -ldl

ifeq ($(MFEM_USE_MPI),YES)
   SIDRE_LIB += -lspio
endif

# OCCA library configuration
ifeq ($(MFEM_USE_OCCA),YES)
  ifndef OCCA_DIR
    OCCA_DIR := @MFEM_DIR@/../occa
  endif
  OCCA_OPT := -I$(OCCA_DIR)/include
  OCCA_LIB := -L$(OCCA_DIR)/lib -locca
endif

#Acrotensor library configs
ifeq ($(MFEM_USE_ACROTENSOR),YES)
   ifndef CUDA_DIR
      CUDA_DIR = /usr/local/cuda
   endif
   ACROTENSOR_DIR = @MFEM_DIR@/../acrotensor
   ACROTENSOR_OPT = -I$(ACROTENSOR_DIR)/inc -I$(CUDA_DIR)/include -std=c++11 -DACRO_HAVE_CUDA
   ACROTENSOR_LIB = -L$(ACROTENSOR_DIR)/lib -L$(CUDA_DIR)/lib64 -lacrotensor -lcuda -lcudart -lnvrtc
endif

# If YES, enable some informational messages
VERBOSE = NO

# Optional build tag
MFEM_BUILD_TAG = $(shell uname -snm)
