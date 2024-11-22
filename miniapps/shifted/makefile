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
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/shifted/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

DIFFUSION_SRC = diffusion.cpp sbm_solver.cpp marking.cpp
DIFFUSION_OBJ = $(DIFFUSION_SRC:.cpp=.o)
DISTANCE_SRC = distance.cpp
DISTANCE_OBJ = $(DISTANCE_SRC:.cpp=.o)
EXTRAPOLATE_SRC = extrapolate.cpp extrapolator.cpp marking.cpp
EXTRAPOLATE_OBJ = $(EXTRAPOLATE_SRC:.cpp=.o)
ALGOIM_SRC =  lsf_integral.cpp
ALGOIM_OBJ = $(ALGOIM_SRC:.cpp=.o)

PAR_MINIAPPS = distance diffusion extrapolate lsf_integral

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = 
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

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

%.o: $(SRC)%.cpp $(SRC)sbm_aux.hpp $(wildcard $(SRC)%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

lsf_integral: $(ALGOIM_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(ALGOIM_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

distance: $(DISTANCE_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(DISTANCE_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

diffusion: $(DIFFUSION_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(DIFFUSION_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

extrapolate: $(EXTRAPOLATE_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(EXTRAPOLATE_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := Shifted miniapp
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
	rm -f *.o *~ distance diffusion extrapolate lsf_integral 
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f diffusion.mesh diffusion.gf
	@rm -rf ParaViewDistance ParaViewDiffusion ParaViewExtrapolate ParaViewLSF
