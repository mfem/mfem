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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/sundials/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex9 ex10 ex16
PAR_EXAMPLES = ex9p ex10p ex16p
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

ifeq ($(MFEM_USE_SUNDIALS),NO)
$(EXAMPLES):
	$(error MFEM is not configured with SUNDIALS)
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
SERIAL_NAME := Serial SUNDIALS example
PARALLEL_NAME := Parallel SUNDIALS example
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(SERIAL_NAME))

# Testing: Specific execution options:
# Example 9: test CVODE with CV_ADAMS (non-stiff implicit) time stepping
EX9_COMMON_ARGS := -m ../../data/periodic-hexagon.mesh -p 0 -s 7
EX9_ARGS  := $(EX9_COMMON_ARGS) -r  2 -dt 0.0018 -vs 25
EX9P_ARGS := $(EX9_COMMON_ARGS) -rp 1 -dt 0.0009 -vs 50
ex9-test-seq: ex9
	@$(call mfem-test,$<,, $(SERIAL_NAME),$(EX9_ARGS))
ex9p-test-par: ex9p
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME),$(EX9P_ARGS))
# Example 10: test CVODE with CV_BDF (stiff implicit) time stepping
EX10_COMMON_ARGS := -m ../../data/beam-quad.mesh -o 2 -s 5 -dt 0.15 -tf 6 -vs 10
EX10_ARGS  := $(EX10_COMMON_ARGS) -r  2
EX10P_ARGS := $(EX10_COMMON_ARGS) -rp 1
ex10-test-seq: ex10
	@$(call mfem-test,$<,, $(SERIAL_NAME),$(EX10_ARGS))
ex10p-test-par: ex10p
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME),$(EX10P_ARGS))

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
