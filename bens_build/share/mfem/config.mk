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

# Variables corresponding to defines in config.hpp (YES, NO, or value)
MFEM_VERSION           = 40001
MFEM_VERSION_STRING    = 4.0.1
MFEM_SOURCE_DIR        = /g/g19/bs/quartz/mfem
MFEM_INSTALL_DIR       = /g/g19/bs/quartz/mfem/bens_build
MFEM_GIT_STRING        = heads/AIR-0-g771bfddaa2e09759daf9bf7982aed2c9c99c9681
MFEM_USE_MPI           = YES
MFEM_USE_METIS         = YES
MFEM_USE_METIS_5       = YES
MFEM_DEBUG             = NO
MFEM_USE_EXCEPTIONS    = NO
MFEM_USE_GZSTREAM      = NO
MFEM_USE_LIBUNWIND     = NO
MFEM_USE_LAPACK        = NO
MFEM_THREAD_SAFE       = NO
MFEM_USE_LEGACY_OPENMP = NO
MFEM_USE_OPENMP        = NO
MFEM_USE_MEMALLOC      = YES
MFEM_TIMER_TYPE        = 2
MFEM_USE_SUNDIALS      = NO
MFEM_USE_MESQUITE      = NO
MFEM_USE_SUITESPARSE   = NO
MFEM_USE_SUPERLU       = NO
MFEM_USE_STRUMPACK     = NO
MFEM_USE_GECKO         = NO
MFEM_USE_GNUTLS        = NO
MFEM_USE_NETCDF        = NO
MFEM_USE_PETSC         = NO
MFEM_USE_MPFR          = NO
MFEM_USE_SIDRE         = NO
MFEM_USE_CONDUIT       = NO
MFEM_USE_PUMI          = NO
MFEM_USE_CUDA          = NO
MFEM_USE_HIP           = 
MFEM_USE_RAJA          = NO
MFEM_USE_OCCA          = NO

# Compiler, compile options, and link options
MFEM_CXX       = /usr/tce/packages/mvapich2/mvapich2-2.3-intel-18.0.1/bin/mpicxx
MFEM_CPPFLAGS  = 
MFEM_CXXFLAGS  = -O3 -DNDEBUG -std=c++11
MFEM_TPLFLAGS  =  -I/g/g19/bs/quartz/metis/build/Linux-x86_64/include -I/g/g19/bs/quartz/hypre/src/hypre/include
MFEM_INCFLAGS  = -I$(MFEM_INC_DIR) $(MFEM_TPLFLAGS)
MFEM_PICFLAG   = 
MFEM_FLAGS     = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_INCFLAGS)
MFEM_EXT_LIBS  =  /g/g19/bs/quartz/metis/build/Linux-x86_64/lib/libmetis.a /g/g19/bs/quartz/hypre/src/hypre/lib/libHYPRE.a
MFEM_LIBS      = -L$(MFEM_LIB_DIR) -lmfem $(MFEM_EXT_LIBS)
MFEM_LIB_FILE  = $(MFEM_LIB_DIR)/libmfem.a
MFEM_STATIC    = YES
MFEM_SHARED    = NO
MFEM_BUILD_TAG = Linux-3.10.0-957.5.1.3chaos.ch6.x86_64
MFEM_PREFIX    = /g/g19/bs/quartz/mfem/bens_build
MFEM_INC_DIR   = /g/g19/bs/quartz/mfem/bens_build/include
MFEM_LIB_DIR   = /g/g19/bs/quartz/mfem/bens_build/lib

# Location of test.mk
MFEM_TEST_MK = /g/g19/bs/quartz/mfem/bens_build/share/mfem/test.mk

# Command used to launch MPI jobs
MFEM_MPIEXEC    = /usr/bin/srun
MFEM_MPIEXEC_NP = -np
MFEM_MPI_NP     = 4

# The NVCC compiler cannot link with -x=cu
MFEM_LINK_FLAGS := $(filter-out -x=cu, $(MFEM_FLAGS))

# Optional extra configuration

