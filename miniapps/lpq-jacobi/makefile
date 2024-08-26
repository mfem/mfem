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

# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

LPQ_COMMON_SRC = lpq-common.cpp
LPQ_COMMON_OBJ = $(LPQ_COMMON_SRC:.cpp=.o)

SEQ_MINIAPPS = # No serial version
PAR_MINIAPPS = lpq-jacobi mg-lpq-jacobi

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
.PHONY: all lib-common copy_meshes clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

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

copy_meshes: $(MESHES_DEST_DIR) $(addprefix $(MESHES_DEST_DIR)/, $(notdir $(SRC_MESH_FILES)))
	@echo "Copying meshing miniapps data files ..."

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common $(MESHES_DEST_DIR)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_LIB) $(MFEM_LIBS)

all: copy_meshes $(MINIAPPS)

lpq-jacobi: lpq-jacobi.cpp lpq-common.cpp lpq-common.hpp
	$(MFEM_CXX) $(MFEM_FLAGS) lpq-jacobi.cpp lpq-common.cpp -o lpq-jacobi $(COMMON_LIB) $(MFEM_LIBS)

mg-lpq-jacobi: mg-lpq-jacobi.cpp lpq-common.cpp lpq-common.hpp
	$(MFEM_CXX) $(MFEM_FLAGS) mg-lpq-jacobi.cpp lpq-common.cpp -o mg-lpq-jacobi $(COMMON_LIB) $(MFEM_LIBS)

abs-l1-partial: abs-l1-partial.cpp lpq-common.cpp lpq-common.hpp
	$(MFEM_CXX) $(MFEM_FLAGS) abs-l1-partial.cpp lpq-common.cpp -o abs-l1-partial $(COMMON_LIB) $(MFEM_LIBS)

# Create the destination directory if it doesn't exist
$(MESHES_DEST_DIR):
	mkdir -p $(MESHES_DEST_DIR)

# Define the rule for copying each mesh file
$(MESHES_DEST_DIR)/%: $(SRC_MESHING_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

$(MESHES_DEST_DIR)/%: $(SRC_GSLIB_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

$(MESHES_DEST_DIR)/%: $(SRC_DATA_DIR)/% | $(MESHES_DEST_DIR)
	cp $< $@

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test-file,$<, $(RUN_MPI), Meshing miniapp,$(<).mesh)
%-test-seq: %
	@$(call mfem-test-file,$<,, Meshing miniapp,$(<).mesh)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

# Clean targets
.PHONY: clean clean-build clean-exec

clean: clean-build clean-exec

clean-build:
	@echo "Cleaning build files..."
	rm -f *.o *~ lpq-jacobi
	rm -f *.o *~ mg-lpq-jacobi
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@echo "Cleaning meshes and CSV files..."
	rm -rf $(MESHES_DEST_DIR)
	rm -f *.csv
