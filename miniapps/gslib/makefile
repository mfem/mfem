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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/gslib/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

ifeq ($(MFEM_USE_GSLIB),YES)
   SEQ_MINIAPPS = findpts field-diff field-interp schwarz_ex1
   PAR_MINIAPPS = pfindpts schwarz_ex1p
else
   SEQ_MINIAPPS =
   PAR_MINIAPPS =
endif

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_LIB) $(MFEM_LIBS)

all: $(MINIAPPS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

# For out-of-source builds, link the data files from the source tree:
ifneq ($(SRC),)
DATA_FILES = triple-pt-1.mesh triple-pt-2.mesh triple-pt-1.gf triple-pt-2.gf
$(DATA_FILES): %: $(SRC)%
	ln -sf $(<) .
field-diff field-interp: | $(DATA_FILES)
.PHONY: copy-data
copy-data: | $(DATA_FILES)
# For out-of-source builds, the test and sample runs for 'field-interp' need
# data from the meshing miniapps directory:
field-interp: | meshing-data
.PHONY: meshing-data
meshing-data:
	$(MAKE) -C ../meshing copy-data
endif

ifeq ($(MFEM_USE_GSLIB),NO)
$(MINIAPPS):
	$(error MFEM is not configured with GSLIB)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := GSLIB miniapp
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TEST_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f interpolated.gf
