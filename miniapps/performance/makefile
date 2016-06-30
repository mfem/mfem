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
MFEM_DIR = ../..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk
# Use the MFEM install directory
# MFEM_DIR = ../../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Compiler specific optimizations.
# For best performance, GCC 5 (or newer) is recommended.
ifneq (,$(findstring $(MFEM_CXX),g++ mpicxx))
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -std=c++03
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   MFEM_CXXFLAGS += --param max-completely-peel-times=3
   # MFEM_CXXFLAGS += -fdump-tree-optimized-blocks
endif

ifeq ($(MFEM_CXX),clang++)
   MFEM_CXXFLAGS += -march=native
   # MFEM_CXXFLAGS += -std=c++03
   MFEM_CXXFLAGS += -std=c++11
   MFEM_CXXFLAGS += -pedantic -Wall

   MFEM_CXXFLAGS += -fcolor-diagnostics
   MFEM_CXXFLAGS += -fvectorize
   MFEM_CXXFLAGS += -fslp-vectorize
   MFEM_CXXFLAGS += -fslp-vectorize-aggressive
   MFEM_CXXFLAGS += -ffp-contract=fast
endif

SEQ_MINIAPPS = ex1
PAR_MINIAPPS = ex1p
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(TEST_MK)

# Testing: Parallel vs. serial runs
%-test-par: %
	@$(call mfem-test,$<, mpirun -np 4, Performance miniapp)
%-test-seq: %
	@$(call mfem-test,$<,, Performance miniapp)

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
