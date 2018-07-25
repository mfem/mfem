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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/gslib_findpts/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = exs
PAR_EXAMPLES = exp
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) 
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

ifeq ($(MFEM_USE_GSLIB_FPT),NO)
$(EXAMPLES):
	$(error MFEM is not configured with GSLIB_FPT)
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
SERIAL_NAME := Serial GSLIB_FPT example
PARALLEL_NAME := Parallel GSLIB_FPT example
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(SERIAL_NAME))

# Testing: Specific execution options:
# Example 9: test explicit CVODE time stepping
EXS_COMMON_ARGS := -m ../../data/periodic-hexagon.mesh -o 1 -qo 3 
EXS_ARGS  := $(EX9_COMMON_ARGS)
EXP_ARGS := $(EX9_COMMON_ARGS)
exs-test-seq: exs
	@$(call mfem-test,$<,, $(SERIAL_NAME),$(EXS_ARGS))
exp-test-par: exp
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME),$(EXP_ARGS))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.* Example9*
	@rm -f deformed.* velocity.* elastic_energy.*
	@rm -f ex16.mesh ex16-mesh.* ex16-init.* ex16-final.* Example16*
