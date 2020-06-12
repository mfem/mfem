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

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/blocksolvers/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

BLOCK_COMMON_SRC = div_free_solver.cpp
BLOCK_COMMON_OBJ = $(BLOCK_COMMON_SRC:.cpp=.o)

PAR_MINIAPPS = mixed_solver_comparison

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

#COMMON_O=../common/pfem_extras.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.cpp $(BLOCK_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(BLOCK_COMMON_OBJ) $(MFEM_LIBS)

%.o: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

PAR_TEST_EXECS = block_solvers_comparison
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
mixed-solver-test-par: block_solvers_comparison
	@$(call mfem-test,$<, $(RUN_MPI), -r 3 -GMG)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf block_solvers_comparison*
