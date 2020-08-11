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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/toys/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = automata life mandel rubik snake lissajous mondrian
PAR_MINIAPPS =
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

# Toys that depend on lib-common
snake rubik: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $@.o $(COMMON_LIB) $(MFEM_LIBS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

# Rules to link the *.pgm file(s) - needed for running the tests/sample runs
# when building out-of-source:
ifneq ($(SRC),)
IMAGE_FILES = $(notdir $(wildcard $(SRC)*.pgm))
$(IMAGE_FILES): %: $(SRC)%
	ln -sf $(<) .
mondrian: | $(IMAGE_FILES)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test-file,$<, $(RUN_MPI), Toys miniapp,$(<).mesh)
%-test-seq: %
	@$(call mfem-test-file,$<,, Toys miniapp,$(<).mesh)
snake-test-seq: snake
	@$(call mfem-test-file,$<,, Toys miniapp,$(<)-init.mesh)
rubik-test-seq: rubik
	@$(call mfem-test-file,$<,, Toys miniapp,$(<)-init.mesh)

# Testing: Specific execution options

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f snake-init.mesh snake-user.mesh snake-joined.mesh snake-c*.mesh
	@rm -f automata.gf automata.mesh rubik-init.mesh mandel.mesh
	@rm -f life.gf life.mesh lissajous.mesh lissajous.gf mondrian.mesh
