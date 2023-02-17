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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/curlcurl/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
OBJ = CurlCurlIntegrator.o
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

PAR_MINIAPPS = curlcurl periodic2D
MINIAPPS = $(PAR_MINIAPPS)

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK) 
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(OBJ) $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

all: $(MINIAPPS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

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

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ 
	rm -f curlcurl periodic2D
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f curlcurl mesh.* sol.*
