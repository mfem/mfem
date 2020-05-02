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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/electromagnetics/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS =
PAR_MINIAPPS = volta tesla maxwell joule
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

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
%: $(SRC)%.cpp %_solver.o $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $@.o $@_solver.o $(COMMON_LIB) \
	  $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(addsuffix _solver.o,$(MINIAPPS)): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

# Rules to copy the *.mesh files - needed for running the sample runs when
# building out-of-source:
ifneq ($(SRC),)
JOULE_MESH_FILES = cylinder-hex.mesh cylinder-tet.mesh
ifeq ($(MFEM_USE_NETCDF),YES)
   JOULE_MESH_FILES += cylinder-hex-q2.gen cylinder-tet-p2.gen
endif
TESLA_MESH_FILES = square-angled-pipe.mesh
VOLTA_MESH_FILES = llnl.mesh
$(JOULE_MESH_FILES) $(TESLA_MESH_FILES) $(VOLTA_MESH_FILES): %: $(SRC)%
	ln -sf $(<) .
joule: | $(JOULE_MESH_FILES)
tesla: | $(TESLA_MESH_FILES)
volta: | $(VOLTA_MESH_FILES)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
volta-test-par: volta-test-1 volta-test-2
volta-test-1: volta
	@$(call mfem-test,$<, $(RUN_MPI), Electromagnetic miniapp,\
	-maxit 2 -dbcs 1 -dbcg -ds '0.0 0.0 0.0 0.2 8.0')
volta-test-2: volta
	@$(call mfem-test,$<, $(RUN_MPI), Electromagnetic miniapp,\
	-maxit 2 -m ../../data/square-disc.mesh \
	-dbcs '1 2 3 4 5 6 7 8' -dbcv '0 0 0 0 1 1 1 1')
tesla-test-par: tesla
	@$(call mfem-test,$<, $(RUN_MPI), Electromagnetic miniapp,\
	-maxit 2 -cr '0 0 -0.2 0 0 0.2 0.2 0.4 1')
maxwell-test-par: maxwell
	@$(call mfem-test,$<, $(RUN_MPI), Electromagnetic miniapp,\
	-abcs '-1' -dp '-0.3 0.0 0.0 0.3 0.0 0.0 0.1 1 .5 .5')
joule-test-par: joule
	@$(call mfem-test,$<, $(RUN_MPI), Electromagnetic miniapp,\
	-m cylinder-hex.mesh -p rod -tf 3)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf Volta-AMR* Tesla-AMR* Maxwell-Parallel* Joule_*
