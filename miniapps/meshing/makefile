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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/meshing/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = mobius-strip klein-bottle toroid trimmer twist mesh-explorer\
	shaper extruder mesh-optimizer minimal-surface polar-nc reflector\
	ref321 mesh-quality hpref
PAR_MINIAPPS = pmesh-optimizer pminimal-surface pmesh-fitting fit-node-position\
	phpref mesh-bounding-boxes
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_LIB) $(MFEM_LIBS)

all: $(MINIAPPS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

# Rules to copy the *.mesh files - needed for running the sample runs when
# building out-of-source:
ifneq ($(SRC),)
MESH_FILES = amr-quad-q2.mesh blade.mesh cube.mesh cube-tet.mesh icf.mesh\
jagged.mesh square01.mesh square01-tri.mesh stretched2D.mesh
$(MESH_FILES): %: $(SRC)%
	ln -sf $(<) .
mesh-optimizer pmesh-optimizer pmesh-fitting fit-node-position: | $(MESH_FILES)
.PHONY: copy-data
copy-data: | $(MESH_FILES)
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
twist-test-seq: twist
	@$(call mfem-test-file,$<,, Meshing miniapp,$(<)-hex-o3-s2-p.mesh)
mesh-optimizer-test-seq: mesh-optimizer
	@$(call mfem-test,$<,, Meshing miniapp)
pmesh-optimizer-test-par: pmesh-optimizer
	@$(call mfem-test,$<, $(RUN_MPI), Parallel meshing miniapp)
mesh-quality-test-seq: mesh-quality
	@$(call mfem-test,$<,, Mesh quality miniapp)
pmesh-fitting-test-par: pmesh-fitting
	@$(call mfem-test,$<, $(RUN_MPI), Parallel mesh fitting miniapp)
fit-node-position-test-par: fit-node-position
	@$(call mfem-test,$<, $(RUN_MPI), Parallel position fitting miniapp)
minimal-surface-test-seq: minimal-surface
	@$(call mfem-test,$<,, Meshing miniapp)
pminimal-surface-test-par: pminimal-surface
	@$(call mfem-test,$<, $(RUN_MPI), Parallel meshing miniapp)
reflector-test-seq: reflector
	@$(call mfem-test-file,$<,, Meshing miniapp,reflected.mesh)
hpref-test-seq: hpref
	@$(call mfem-test,$<,, Serial hp-refinement)
phpref-test-par: phpref
	@$(call mfem-test,$<, $(RUN_MPI), Parallel hp-refinement)
mesh-bounding-boxes-test-par: mesh-bounding-boxes
	@$(call mfem-test,$<, $(RUN_MPI), Parallel bounding boxes)
ref321-test-seq: ref321
	@$(call mfem-test-file,$<,, Meshing miniapp,ref321.mesh)

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
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f mobius-strip.mesh klein-bottle.mesh mesh-explorer.mesh*
	@rm -f toroid-*.mesh twist-*.mesh trimmer.mesh reflected.mesh
	@rm -f partitioning.txt shaper.mesh extruder.mesh ref321.mesh sol.gf
	@rm -f optimized* perturbed* polar-nc.mesh
	@rm -rf mesh-explorer-{visit,paraview}*
	@rm -f mesh.* order.* sol.* refined.mesh
	@rm -rf jacobian-determinant-bounds* bounding-box*
