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
MFEM_USE_MPI         = YES
MFEM_USE_METIS_5     = YES
MFEM_DEBUG           = NO
MFEM_USE_GZSTREAM    = NO
MFEM_USE_LIBUNWIND   = NO
MFEM_USE_LAPACK      = YES
MFEM_THREAD_SAFE     = NO
MFEM_USE_OPENMP      = NO
MFEM_USE_MEMALLOC    = YES
MFEM_TIMER_TYPE      = 2
MFEM_USE_SUNDIALS    = NO
MFEM_USE_MESQUITE    = NO
MFEM_USE_SUITESPARSE = YES
MFEM_USE_SUPERLU     = NO
MFEM_USE_GECKO       = NO
MFEM_USE_GNUTLS      = NO
MFEM_USE_NETCDF      = NO
MFEM_USE_PETSC       = NO
MFEM_USE_MPFR        = NO
MFEM_USE_SIDRE       = NO

# Compiler, compile options, and link options
MFEM_CXX       = mpicxx
MFEM_CPPFLAGS  =
MFEM_CXXFLAGS  = -O3 -g -std=c++11
MFEM_TPLFLAGS  =  -I/home/kvoronin/Documents/mfem_glvis_etc/metis-5.1.0/include -I/home/kvoronin/Documents/mfem_glvis_etc//hypre-2.11.1/src/hypre/include   -I/home/kvoronin/Documents/suitesparse/SuiteSparse/include
MFEM_INCFLAGS  = -I$(MFEM_INC_DIR) $(MFEM_TPLFLAGS)
MFEM_FLAGS     = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_INCFLAGS)
MFEM_LIBS      = -L$(MFEM_LIB_DIR) -lmfem -L/home/kvoronin/Documents/mfem_glvis_etc/metis-5.1.0 -lmetis -L/home/kvoronin/Documents/mfem_glvis_etc//hypre-2.11.1/src/hypre/lib -lHYPRE  -L/home/kvoronin/Documents/lapack/lapack-3.6.1 -llapack -lrefblas -lgfortran -L/home/kvoronin/Documents/suitesparse/SuiteSparse/lib -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -L/home/kvoronin/Documents/mfem_glvis_etc/metis-5.1.0 -lmetis -L/home/kvoronin/Documents/lapack/lapack-3.6.1 -llapack -lrefblas -lgfortran -Wl,-rpath=/home/kvoronin/Documents/suitesparse/SuiteSparse/lib -lrt
MFEM_LIB_FILE  = $(MFEM_LIB_DIR)/libmfem.a
MFEM_BUILD_TAG = Linux kvoronin-OptiPlex-790 x86_64
MFEM_PREFIX    = ./mfem
MFEM_INC_DIR   = $(MFEM_DIR)
MFEM_LIB_DIR   = $(MFEM_DIR)

# Command used to launch MPI jobs
MFEM_MPIEXEC    = mpirun
MFEM_MPIEXEC_NP = -np

# Optional extra configuration
