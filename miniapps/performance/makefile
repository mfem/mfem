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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/performance/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Distinguish x86 from PowerPC systems
MFEM_MACHINE ?= $(shell uname -m)

# Choose the switch MFEM_PERF_SW: gcc_x86_64, gcc_ppc64, or clang.
# The value of MFEM_PERF_SW is used to select MFEM_PERF_CXXFLAGS below.
ifneq (,$(MFEM_PERF_SW))
   # Use the value of MFEM_PERF_SW if already defined
else ifneq (,$(filter %clang++ %mpiclang++,$(MFEM_CXX)))
   MFEM_PERF_SW = clang
else ifneq (,$(filter %g++ %mpicxx %mpic++,$(MFEM_CXX)))
   ifeq ($(MFEM_MACHINE),x86_64)
      MFEM_PERF_SW = gcc_x86_64
   else ifneq (,$(findstring ppc64,$(MFEM_MACHINE)))
      MFEM_PERF_SW = gcc_ppc64
   endif
endif

# Compiler specific optimizations.
# For best performance, GCC 5 (or newer) is recommended.

# - GCC extra options:
# MFEM_PERF_CXXFLAGS_gcc_common += -std=c++03
MFEM_PERF_CXXFLAGS_gcc_common += -std=c++11
MFEM_PERF_CXXFLAGS_gcc_common += -pedantic -Wall
MFEM_PERF_CXXFLAGS_gcc_common += --param max-completely-peel-times=3
# MFEM_PERF_CXXFLAGS_gcc_common += -fdump-tree-optimized-blocks
MFEM_PERF_CXXFLAGS_gcc_x86_64 = -march=native $(MFEM_PERF_CXXFLAGS_gcc_common)
MFEM_PERF_CXXFLAGS_gcc_ppc64 = -mcpu=native -mtune=native\
 $(MFEM_PERF_CXXFLAGS_gcc_common)

# - Clang extra options:
MFEM_PERF_CXXFLAGS_clang += -march=native
# MFEM_PERF_CXXFLAGS_clang += -std=c++03
MFEM_PERF_CXXFLAGS_clang += -std=c++11
MFEM_PERF_CXXFLAGS_clang += -pedantic -Wall
MFEM_PERF_CXXFLAGS_clang += -fcolor-diagnostics
MFEM_PERF_CXXFLAGS_clang += -fvectorize
MFEM_PERF_CXXFLAGS_clang += -fslp-vectorize
MFEM_PERF_CXXFLAGS_clang += -fslp-vectorize-aggressive
MFEM_PERF_CXXFLAGS_clang += -ffp-contract=fast

# Choose MFEM_PERF_CXXFLAGS based on MFEM_PERF_SW:
MFEM_PERF_CXXFLAGS = $(MFEM_PERF_CXXFLAGS_$(MFEM_PERF_SW))
# Add MFEM_PERF_CXXFLAGS to MFEM_CXXFLAGS:
MFEM_CXXFLAGS += $(MFEM_PERF_CXXFLAGS)

SEQ_MINIAPPS = ex1
PAR_MINIAPPS = ex1p
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), Performance miniapp,-rs 2)
ex1-test-seq: ex1
	@$(call mfem-test,$<,, Performance miniapp,-r 2)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ ex1 ex1p
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh mesh.* sol.*
