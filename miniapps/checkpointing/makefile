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

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/checkpointing/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = checkpoint-heterogeneous-state checkpoint-mesh-state \
   checkpoint-ode-state
MINIAPPS = $(SEQ_MINIAPPS)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

all: $(MINIAPPS)

checkpoint-heterogeneous-state: $(SRC)heterogeneous-state.cpp \
   $(SRC)checkpoint_demo.hpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

checkpoint-mesh-state: $(SRC)mesh-state.cpp $(SRC)checkpoint_demo.hpp \
   $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

checkpoint-ode-state: $(SRC)ode-state.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

checkpoint-heterogeneous-state-test-seq: checkpoint-heterogeneous-state
	@$(call mfem-test,$<,,Checkpointing miniapp,-n 8 -c 3 -w 2,explicit-options)
checkpoint-mesh-state-test-seq: checkpoint-mesh-state
	@$(call mfem-test,$<,,Checkpointing miniapp,-r 3 -c 2 -no-pv,explicit-options)
checkpoint-ode-state-test-seq: checkpoint-ode-state
	@$(call mfem-test,$<,,Checkpointing miniapp,,explicit-options)

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	rm -rf paraview
