# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/amgx/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex1 ex1_json
PAR_EXAMPLES = ex1p ex1p_json
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

ifeq ($(MFEM_USE_AMGX),NO)
$(EXAMPLES):
	$(error MFEM is not configured with AMGX)
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
SERIAL_NAME := Serial AMGX example
PARALLEL_NAME := Parallel AMGX example
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(SERIAL_NAME))

# Testing: Specific execution options:
# Example 1/1p:
EX1_ARGS  := -amgx
EX1P_ARGS := -amgx
EX1_JSON_ARGS  := --amgx-file multi_gs.json --amgx-solver
EX1P_JSON_ARGS := --amgx-file amg_pcg.json
ex1-json-test-seq: ex1_json
	@$(call mfem-test,$<,, $(SERIAL_NAME),$(EX1_JSON_ARGS))
ex1p-json-test-par: ex1p_json
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME),$(EX1P_JSON_ARGS))
ex1-test-seq: ex1
	@$(call mfem-test,$<,, $(SERIAL_NAME),$(EX1_ARGS))
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME),$(EX1P_ARGS))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not build)

clean: clean-build

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f .logamgx refined.mesh sol.gf mesh.* sol.*
