# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/solvers/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

BLOCK_SOLVERS_SRC = div_free_solver.cpp block-solvers.cpp
BLOCK_SOLVERS_OBJ = $(BLOCK_SOLVERS_SRC:.cpp=.o)

SER_MINIAPPS = lor
PAR_MINIAPPS = block-solvers plor

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SER_MINIAPPS)
else
   MINIAPPS = $(SER_MINIAPPS) $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: $(SRC)%.cpp
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(MFEM_LIBS)

block-solvers: $(BLOCK_SOLVERS_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(BLOCK_SOLVERS_OBJ) $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
block-solvers-test-par: block-solvers-constant block-solvers-anisotropic
block-solvers-constant: block-solvers
	@$(call mfem-test,$<, $(RUN_MPI), BlockSolver miniapp,-r 2)
block-solvers-anisotropic: block-solvers
	@$(call mfem-test,$<, $(RUN_MPI), BlockSolver miniapp,-r 2\
	  -m $(SRC)anisotropic.mesh -c $(SRC)anisotropic.coeff\
	  -eb $(SRC)anisotropic.bdr)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf block-solvers
