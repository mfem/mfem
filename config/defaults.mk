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

# Default options. To replace these, copy this file to user.mk and modify it.


# Some choices below are based on the system name:
SYSNAME = $(shell uname -s)

CXX = g++
MPICXX = mpicxx

OPTIM_FLAGS = -O3
DEBUG_FLAGS = -g -Wall

# Destination location of make install
# PREFIX = $(HOME)/mfem
PREFIX = ./mfem
# Install program
INSTALL = /usr/bin/install

ifneq ($(SYSNAME),Darwin)
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

# MFEM configuration options: YES/NO values, which are exported to config.mk and
# config.hpp. The values below are the defaults for generating the actual values
# in config.mk and config.hpp.
MFEM_USE_MPI         = NO
MFEM_USE_METIS_5     = NO
MFEM_DEBUG           = NO
MFEM_USE_LAPACK      = NO
MFEM_THREAD_SAFE     = NO
MFEM_USE_OPENMP      = NO
MFEM_USE_MEMALLOC    = YES
MFEM_TIMER_TYPE      = $(if $(findstring Darwin,$(SYSNAME)),0,2)
MFEM_USE_MESQUITE    = NO
MFEM_USE_SUITESPARSE = NO
MFEM_USE_SUPERLU     = NO
MFEM_USE_GECKO       = NO
MFEM_USE_GNUTLS      = NO
MFEM_USE_NETCDF      = NO

# HYPRE library configuration (needed to build the parallel version)
HYPRE_DIR = @MFEM_DIR@/../hypre-2.10.0b/src/hypre
HYPRE_OPT = -I$(HYPRE_DIR)/include
HYPRE_LIB = -L$(HYPRE_DIR)/lib -lHYPRE

# METIS library configuration
ifeq ($(MFEM_USE_SUPERLU),NO)
   METIS_DIR ?= @MFEM_DIR@/../metis-4.0
   METIS_OPT ?=
   METIS_LIB ?= -L$(METIS_DIR) -lmetis
   MFEM_USE_METIS_5 ?= NO
else
   # ParMETIS currently needed only with SuperLU
   METIS_DIR ?= @MFEM_DIR@/../parmetis-4.0.3
   METIS_OPT ?=
   METIS_LIB ?= -L$(METIS_DIR) -lparmetis -lmetis
   MFEM_USE_METIS_5 ?= YES
endif

# LAPACK library configuration
LAPACK_OPT =
LAPACK_LIB = -llapack
ifeq ($(SYSNAME),Darwin)
   LAPACK_LIB = -framework Accelerate
endif

# OpenMP configuration
OPENMP_OPT = -fopenmp
OPENMP_LIB =

# Used when MFEM_TIMER_TYPE = 2
POSIX_CLOCKS_LIB = -lrt

# MESQUITE library configuration
MESQUITE_DIR = @MFEM_DIR@/../mesquite-2.99
MESQUITE_OPT = -I$(MESQUITE_DIR)/include
MESQUITE_LIB = -L$(MESQUITE_DIR)/lib -lmesquite

# SuiteSparse library configuration
LIB_RT = -lrt
ifeq ($(SYSNAME),Darwin)
   LIB_RT =
endif
SUITESPARSE_DIR = @MFEM_DIR@/../SuiteSparse
SUITESPARSE_OPT = -I$(SUITESPARSE_DIR)/include
SUITESPARSE_LIB = -L$(SUITESPARSE_DIR)/lib -lklu -lbtf -lumfpack -lcholmod\
 -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig $(LIB_RT) $(METIS_LIB)\
 $(LAPACK_LIB)

# SuperLU library configuration
SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_5.1.0
SUPERLU_OPT = -I$(SUPERLU_DIR)/SRC
SUPERLU_LIB = -L$(SUPERLU_DIR)/SRC -lsuperlu_dist

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

# If YES, enable some informational messages
VERBOSE = NO

# Optional build tag
MFEM_BUILD_TAG = $(shell uname -snm)
