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

STATIC = YES
SHARED = NO

ifneq ($(NOTMAC),)
   AR      = ar
   ARFLAGS = cruv
   RANLIB  = ranlib
   PICFLAG = -fPIC
   SO_EXT  = so
   SO_VER  = so.$(MFEM_VERSION_STRING)
   BUILD_SOFLAGS = -shared -Wl,-soname,libmfem.$(SO_VER)
   BUILD_RPATH = -Wl,-rpath,$(BUILD_REAL_DIR)
   INSTALL_SOFLAGS = $(BUILD_SOFLAGS)
   INSTALL_RPATH = -Wl,-rpath,@MFEM_LIB_DIR@
else
   # Silence "has no symbols" warnings on Mac OS X
   AR      = ar
   ARFLAGS = Scruv
   RANLIB  = ranlib -no_warning_for_no_symbols
   PICFLAG = -fPIC
   SO_EXT  = dylib
   SO_VER  = $(MFEM_VERSION_STRING).dylib
   MAKE_SOFLAGS = -Wl,-dylib,-install_name,$(1)/libmfem.$(SO_VER),\
      -compatibility_version,$(MFEM_VERSION_STRING),\
      -current_version,$(MFEM_VERSION_STRING),\
      -undefined,dynamic_lookup
   BUILD_SOFLAGS = $(subst $1 ,,$(call MAKE_SOFLAGS,$(BUILD_REAL_DIR)))
   BUILD_RPATH = -Wl,-undefined,dynamic_lookup
   INSTALL_SOFLAGS = $(subst $1 ,,$(call MAKE_SOFLAGS,$(MFEM_LIB_DIR)))
   INSTALL_RPATH = -Wl,-undefined,dynamic_lookup
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

MFEM_USE_MPI         = NO
MFEM_USE_METIS       = $(MFEM_USE_MPI)
MFEM_USE_METIS_5     = NO
MFEM_DEBUG           = NO
MFEM_USE_EXCEPTIONS  = NO
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
MFEM_USE_CONDUIT     = NO
MFEM_USE_PUMI        = NO

# Compile and link options for zlib.
ZLIB_DIR =
ZLIB_OPT = $(if $(ZLIB_DIR),-I$(ZLIB_DIR)/include)
ZLIB_LIB = $(if $(ZLIB_DIR),$(ZLIB_RPATH) -L$(ZLIB_DIR)/lib ,)-lz
ZLIB_RPATH = -Wl,-rpath,$(ZLIB_DIR)/lib

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
OPENMP_OPT = -fopenmp
OPENMP_LIB =

# Used when MFEM_TIMER_TYPE = 2
POSIX_CLOCKS_LIB = -lrt

# SUNDIALS library configuration
SUNDIALS_DIR = @MFEM_DIR@/../sundials-3.0.0
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
SUITESPARSE_LIB = -Wl,-rpath,$(SUITESPARSE_DIR)/lib -L$(SUITESPARSE_DIR)/lib\
 -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd\
 -lsuitesparseconfig $(LIB_RT) $(METIS_LIB) $(LAPACK_LIB)

# SuperLU library configuration
SUPERLU_DIR = @MFEM_DIR@/../SuperLU_DIST_5.1.0
SUPERLU_OPT = -I$(SUPERLU_DIR)/SRC
SUPERLU_LIB = -Wl,-rpath,$(SUPERLU_DIR)/SRC -L$(SUPERLU_DIR)/SRC -lsuperlu_dist

# SCOTCH library configuration (required by STRUMPACK <= v2.1.0, optional in
# STRUMPACK >= v2.2.0)
SCOTCH_DIR = @MFEM_DIR@/../scotch_6.0.4
SCOTCH_OPT = -I$(SCOTCH_DIR)/include
SCOTCH_LIB = -L$(SCOTCH_DIR)/lib -lptscotch -lptscotcherr -lscotch -lscotcherr\
 -lpthread

# SCALAPACK library configuration (required by STRUMPACK)
SCALAPACK_DIR = @MFEM_DIR@/../scalapack-2.0.2
SCALAPACK_OPT = -I$(SCALAPACK_DIR)/SRC
SCALAPACK_LIB = -L$(SCALAPACK_DIR)/lib -lscalapack $(LAPACK_LIB)

# MPI Fortran library, needed e.g. by STRUMPACK
# MPICH:
MPI_FORTRAN_LIB = -lmpifort
# OpenMPI:
# MPI_FORTRAN_LIB = -lmpi_mpifh
# Additional Fortan library:
# MPI_FORTRAN_LIB += -lgfortran

# STRUMPACK library configuration
STRUMPACK_DIR = @MFEM_DIR@/../STRUMPACK-build
STRUMPACK_OPT = -I$(STRUMPACK_DIR)/include $(SCOTCH_OPT)
# If STRUMPACK was build with OpenMP support, the following may be need:
# STRUMPACK_OPT += $(OPENMP_OPT)
STRUMPACK_LIB = -L$(STRUMPACK_DIR)/lib -lstrumpack $(MPI_FORTRAN_LIB)\
 $(SCOTCH_LIB) $(SCALAPACK_LIB)

# Gecko library configuration
GECKO_DIR = @MFEM_DIR@/../gecko
GECKO_OPT = -I$(GECKO_DIR)/inc
GECKO_LIB = -L$(GECKO_DIR)/lib -lgecko

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
   -lsidre -lslic -laxom_utils -lconduit -lconduit_relay -lhdf5 $(ZLIB_LIB) -ldl

# PUMI
# Note that PUMI_DIR is needed -- it is used to check for gmi_sim.h
PUMI_DIR = @MFEM_DIR@/../pumi-2.1.0
PUMI_OPT = -I$(PUMI_DIR)/include
PUMI_LIB = -L$(PUMI_DIR)/lib -lpumi -lcrv -lma -lmds -lapf -lpcu -lgmi -lparma\
   -llion -lmth -lapf_zoltan -lspr

# If YES, enable some informational messages
VERBOSE = NO

# Optional build tag
MFEM_BUILD_TAG = $(shell uname -snm)
