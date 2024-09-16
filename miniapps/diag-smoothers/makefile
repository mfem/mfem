# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/lpq-jacobi/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk

MFEM_LIB_FILE = mfem_is_not_built

-include $(DEFAULTS_MK)
-include $(CONFIG_MK)

LPQ_COMMON_SRC = lpq-common.cpp
LPQ_COMMON_OBJ = $(LPQ_COMMON_SRC:.cpp=.o)

PAR_MINIAPPS = lpq-jacobi mg-lpq-jacobi abs-l1-jacobi mg-abs-l1-jacobi

MINIAPPS = $(if $(MFEM_USE_MPI:NO=),$(PAR_MINIAPPS),)

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

# Mesh directories and files
# Copy meshes into ./meshes
SRC_MESHING_DIR = ../meshing
SRC_GSLIB_DIR = ../gslib
SRC_DATA_DIR = ../../data
MESHES_DEST_DIR = ./meshes

MESH_FILES_FROM_MESHING = icf.mesh cube.mesh
MESH_FILES_FROM_GSLIB = triple-pt-1.mesh triple-pt-2.mesh
MESH_FILES_FROM_DATA = beam-tet.mesh square-disc-p2.mesh fichera-mixed-p2.mesh \
					   amr-quad.mesh ref-cube.mesh ref-square.mesh

SRC_MESH_FILES = $(addprefix $(SRC_MESHING_DIR)/, $(MESH_FILES_FROM_MESHING)) \
                 $(addprefix $(SRC_GSLIB_DIR)/, $(MESH_FILES_FROM_GSLIB)) \
                 $(addprefix $(SRC_DATA_DIR)/, $(MESH_FILES_FROM_DATA))

# Phony targets
.PHONY: all lib-common copy_meshes clean clean-build clean-exec test

# Main targets
all: copy_meshes $(MINIAPPS)

# Build rules
$(MINIAPPS): %: %.cpp $(LPQ_COMMON_OBJ) lpq-common.hpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) $< $(LPQ_COMMON_OBJ) -o $@ $(COMMON_LIB) $(MFEM_LIBS)

$(LPQ_COMMON_OBJ): %.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# Library and mesh targets
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

copy_meshes: $(MESHES_DEST_DIR) $(addprefix $(MESHES_DEST_DIR)/, $(notdir $(SRC_MESH_FILES)))
	@echo "Copying meshing miniapps data files ..."

# Create the destination directory if it doesn't exist
$(MESHES_DEST_DIR):
	mkdir -p $(MESHES_DEST_DIR)

$(MESHES_DEST_DIR)/%: $(SRC_MESHING_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

$(MESHES_DEST_DIR)/%: $(SRC_GSLIB_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

$(MESHES_DEST_DIR)/%: $(SRC_DATA_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

# Test targets
MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

lpq-jacobi-test-par: lpq-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), Lpq-Jqcobi miniapp, \
		-m meshes/cube.mesh \
		-rs 1 -rp 1 \
		-s 1 -i 1 \
		-p 1.5 -q 0.75 \
		-pc 1 \
		-no-mon)

abs-l1-jacobi-test-par: abs-l1-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), Abs-value-L1-Jqcobi miniapp, \
		-m meshes/cube.mesh \
		-rs 1 -rp 1 \
		-s 1 -i 1 -a 4 \
		-pc 1 \
		-no-mon)

mg-lpq-jacobi-test-par: mg-lpq-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), MG Lpq-Jqcobi miniapp, \
		-m meshes/cube.mesh \
		-rs 1 -rp 1 \
		-ol 1 -gl 1 \
		-s 1 -i 1 \
		-p 1.5 -q 0.75 \
		-no-mon)

mg-abs-l1-jacobi-test-par: mg-abs-l1-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), MG Abs-value-L1-Jqcobi miniapp, \
		-m meshes/cube.mesh \
		-rs 1 -rp 1\
		-ol 1 -gl 1 \
		-s 1 -i 1 -a 4 \
		-no-mon)

# Clean targets
clean: clean-build clean-exec
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common clean

clean-build:
	rm -f *.o *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	rm -rf $(MESHES_DEST_DIR)
	rm -f *.csv

# Error handling
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)
