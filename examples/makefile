# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

CC = g++
OPTS = -O3
DEBUG_OPTS = -g -DDEBUG

MFEM_DIR = ..
MFEM_LIB = -L$(MFEM_DIR) -lmfem

# Link with LAPACK? (needed if MFEM was compiled with LAPACK support)
USE_LAPACK = NO

# LAPACK and BLAS
LAPACK_DIR = $(HOME)/lapack
LAPACK_LIB = -L$(LAPACK_DIR) -llapack
BLAS_DIR   = $(HOME)/lapack
BLAS_LIB   = -L$(LAPACK_DIR) -lblas -lgfortran
# on a Mac:
# BLAS_LIB   = -L$(LAPACK_DIR) -lblas
LAPACK_LIBS_NO  =
LAPACK_LIBS_YES = $(LAPACK_LIB) $(BLAS_LIB)
LAPACK_LIBS     = $(LAPACK_LIBS_$(USE_LAPACK))

LIBS = $(MFEM_LIB) $(LAPACK_LIBS)

all: ex1

ex1: ex1.cpp
	$(CC) $(OPTS) -I$(MFEM_DIR) ex1.cpp -o ex1 $(LIBS)

debug:
	make "OPTS=$(DEBUG_OPTS)"

clean:
	rm -f *.o ex1 sol.gf *~
