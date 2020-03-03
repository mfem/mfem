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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/meshing/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = mobius-strip klein-bottle toroid \
	mesh-explorer shaper extruder mesh-optimizer
PAR_MINIAPPS = pmesh-optimizer
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

# Rules to copy the *.mesh files - needed for running the sample runs when
# building out-of-source:
ifneq ($(SRC),)
MESH_FILES = blade.mesh icf.mesh
$(MESH_FILES): %: $(SRC)%
	ln -sf $(<) .
mesh-optimizer pmesh-optimizer: | $(MESH_FILES)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test-file,$<, $(RUN_MPI), Meshing miniapp,$(<).mesh)
%-test-seq: %
	@$(call mfem-test-file,$<,, Meshing miniapp,$(<).mesh)
toroid-test-seq: toroid
	@$(call mfem-test-file,$<,, Meshing miniapp,$(<)-wedge-o3-s0.mesh)
mesh-optimizer-test-seq: mesh-optimizer
	@$(call mfem-test,$<,, Meshing miniapp)
pmesh-optimizer-test-par: pmesh-optimizer
	@$(call mfem-test,$<, $(RUN_MPI), Parallel meshing miniapp)

# Testing: Specific execution options
mesh-explorer-test-seq:
	@true
shaper-test-seq:
	@true

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ mobius-strip klein-bottle toroid
	rm -f mesh-explorer shaper extruder
	rm -f mesh-optimizer pmesh-optimizer
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f mobius-strip.mesh klein-bottle.mesh mesh-explorer.mesh
	@rm -f toroid-*.mesh
	@rm -f partitioning.txt shaper.mesh extruder.mesh
	@rm -f optimized* perturbed*
