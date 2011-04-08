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

# Serial compiler
CC = g++
OPTS = -O3 -I$(MFEM_DIR)
DEBUG_OPTS = -g -I$(MFEM_DIR)
LIBS = $(MFEM_LIB) $(LAPACK_LIBS)

# Parallel compiler
MPICC = mpiCC
MPIOPTS = $(OPTS) -I$(HYPRE_DIR)/include
MPIDEBUG_OPTS = $(DEBUG_OPTS) -I$(HYPRE_DIR)/include
MPILIBS = $(LIBS) -L$(METIS_DIR) -lmetis -L$(HYPRE_DIR)/lib -lHYPRE

# The MFEM library
MFEM_DIR = ..
MFEM_LIB = -L$(MFEM_DIR) -lmfem

# The METIS and HYPRE libraries (needed for the parallel examples)
METIS_DIR = ../../metis-4.0
HYPRE_DIR = ../../hypre-2.7.0b/src/hypre

# The LAPACK and BLAS libraries (needed if MFEM was compiled with LAPACK support)
USE_LAPACK = YES
LAPACK_DIR = $(HOME)/lapack
LAPACK_LIB = -L$(LAPACK_DIR) -llapack
BLAS_DIR   = $(HOME)/lapack
BLAS_LIB   = -L$(LAPACK_DIR) -lblas -lgfortran
# on a Mac:
# BLAS_LIB   = -L$(LAPACK_DIR) -lblas
LAPACK_LIBS_NO  =
LAPACK_LIBS_YES = $(LAPACK_LIB) $(BLAS_LIB)
LAPACK_LIBS     = $(LAPACK_LIBS_$(USE_LAPACK))

serial: ex1 ex2 ex3

parallel: ex1p ex2p ex3p

ex1: ex1.cpp
	$(CC) $(OPTS) ex1.cpp -o ex1 $(LIBS)

ex1p: ex1p.cpp
	$(MPICC) $(MPIOPTS) ex1p.cpp -o ex1p $(MPILIBS)

ex2: ex2.cpp
	$(CC) $(OPTS) ex2.cpp -o ex2 $(LIBS)

ex2p: ex2p.cpp
	$(MPICC) $(MPIOPTS) ex2p.cpp -o ex2p $(MPILIBS)

ex3: ex3.cpp
	$(CC) $(OPTS) ex3.cpp -o ex3 $(LIBS)

ex3p: ex3p.cpp
	$(MPICC) $(MPIOPTS) ex3p.cpp -o ex3p $(MPILIBS)

debug:
	make "OPTS=$(DEBUG_OPTS)" serial

pdebug:
	make "MPIOPTS=$(MPIDEBUG_OPTS)" parallel

clean:
	rm -f *.o *~ ex1 ex1p ex2 ex2p ex3 ex3p
	rm -f refined.mesh displaced.mesh sol.gf
