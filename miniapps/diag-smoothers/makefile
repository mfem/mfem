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

SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/diag-smoothers/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk

MFEM_LIB_FILE = mfem_is_not_built

-include $(DEFAULTS_MK)
-include $(CONFIG_MK)

DS_COMMON_SRC = ds-common.cpp
DS_COMMON_OBJ = $(DS_COMMON_SRC:.cpp=.o)

PAR_MINIAPPS = abs-l1-jacobi mg-abs-l1-jacobi

MINIAPPS = $(if $(MFEM_USE_MPI:NO=),$(PAR_MINIAPPS),)

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

# Phony targets
.PHONY: all lib-common clean clean-build clean-exec test

# Main targets
all: $(MINIAPPS)

# Build rules
$(MINIAPPS): %: %.cpp $(DS_COMMON_OBJ) ds-common.hpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) $< $(DS_COMMON_OBJ) -o $@ $(COMMON_LIB) $(MFEM_LIBS)

$(DS_COMMON_OBJ): %.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# Library and mesh targets
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

# Test targets
MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

abs-l1-jacobi-test-par: abs-l1-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), Abs-value-L1-Jqcobi miniapp, \
		-m ../../data/ref-cube.mesh \
		-rs 1 -rp 1 \
		-s 1 -i 1 -a 4 \
		-pc 1 \
		-no-mon)

mg-abs-l1-jacobi-test-par: mg-abs-l1-jacobi
	@$(call mfem-test,$<, $(RUN_MPI), MG Abs-value-L1-Jqcobi miniapp, \
		-m ../../data/ref-cube.mesh \
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
	@true

# Error handling
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)
