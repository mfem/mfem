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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/occa/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex0 ex0v ex1 ex16
PAR_EXAMPLES = ex1p
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

all: $(EXAMPLES) cache-kernels

ifeq ($(MFEM_USE_OCCA),NO)
$(EXAMPLES):
	$(error MFEM is not configured with OCCA)
endif

MFEM_TESTS = EXAMPLES
include $(TEST_MK)

# Testing: Parallel vs. serial runs
%-test-par: %
	@$(call mfem-test,$<, mpirun -np 4, Parallel example)
%-test-seq: %
	@$(call mfem-test,$<,, Serial example)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

#---[ OCCA ]----------------------------
MFEM_REAL_DIR      := $(realpath $(MFEM_DIR))
OCCA_CACHE_DIR     ?= ${HOME}/.occa
OCCA_LIB_CACHE_DIR := $(OCCA_CACHE_DIR)/libraries

OKL_KERNELS        := $(realpath $(shell find $(MFEM_REAL_DIR) -type f -name '*.okl'))
OKL_CACHED_KERNELS := $(subst kernels/,,$(subst $(MFEM_REAL_DIR)/,$(OCCA_LIB_CACHE_DIR)/mfem/,$(OKL_KERNELS)))

# Cache kernels in the OCCA cache directory
.PHONY: cache-kernels
cache-kernels: $(OKL_CACHED_KERNELS)

.PHONY: clear-cache
clear-kernels: clear-mfem-kernels $(OKL_CACHED_KERNELS)

.PHONY: clear-cache
clear-mfem-kernels:
	@echo y | occa clear -l mfem

$(OCCA_LIB_CACHE_DIR)/mfem/fem/%.okl: $(MFEM_REAL_DIR)/fem/kernels/%.okl
	@echo "Caching: $(subst $(MFEM_REAL_DIR)/,,$<)"
	@occa cache mfem/$(subst $(OCCA_LIB_CACHE_DIR)/mfem/,,$(dir $@)) $<

$(OCCA_LIB_CACHE_DIR)/mfem/linalg/%.okl: $(MFEM_REAL_DIR)/linalg/kernels/%.okl
	@echo "Caching: $(subst $(MFEM_REAL_DIR)/,,$<)"
	@occa cache mfem/$(subst $(OCCA_LIB_CACHE_DIR)/mfem/,,$(dir $@)) $<
#=======================================
