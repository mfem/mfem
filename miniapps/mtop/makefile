# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mtop/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MTOP_COMMON_SRC = mtop_solvers.cpp linear_elasticity_solver.cpp

MTOP_COMMON_OBJ = $(MTOP_COMMON_SRC:.cpp=.o)

SEQ_MINIAPPS = elasticity_pa_regression
MTOP_MPI_MINIAPPS = mtop_test_iso_elasticity mtop_ex_linear_elasticity_solver
MINIAPPS = $(SEQ_MINIAPPS)

# The first explicit rule below is for an MPI executable. Ensure an invocation
# without a target still selects the serial-safe aggregate target.
.DEFAULT_GOAL := all

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all parallel clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

$(MTOP_MPI_MINIAPPS): %: %.o $(MTOP_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

$(SEQ_MINIAPPS): %: %.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# The mtop solver drivers require an MPI-enabled MFEM build. Keep them out of
# the default target so a stale external make configuration cannot select them
# while linking against a serial MFEM library.
parallel: $(MTOP_MPI_MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := MTop miniapp
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TEST_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME),,NO_VIS_OPTION)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(MTOP_MPI_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf SeqHeat* ParHeat*
