# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
              checkpoint-forward-euler checkpoint-backward-euler
MINIAPPS = $(SEQ_MINIAPPS)

.SUFFIXES:
.PHONY: all clean clean-build clean-exec

all: $(MINIAPPS)

$(SEQ_MINIAPPS): checkpoint-%: $(SRC)%.cpp $(SRC)checkpoint_demo.hpp \
   $(SRC)euler_checkpoint.hpp \
   $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Three independent recipe lines preserve mfem-test's per-case exit checks.
define checkpoint-test
@$(call mfem-test,$(1),$(2),$(3),$(4),$(5))
@$(call mfem-test,$(1),$(2),$(3),$(4) --controller,$(5)-controller)
@$(call mfem-test,$(1),$(2),$(3),$(4) --controller --window-size 2,$(5)-controller-window)
endef

checkpoint-heterogeneous-state-test-seq: checkpoint-heterogeneous-state
	$(call checkpoint-test,$<,,Checkpointing miniapp,-n 8,growing-storage)
	$(call checkpoint-test,$<,,Checkpointing miniapp,-n 8 -m 9,bounded-storage)

checkpoint-forward-euler-test-seq checkpoint-backward-euler-test-seq: \
   %-test-seq: %
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage memory-block,memory-block)
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage file-block,file-block)
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage memory-snapshots,memory-snapshots)
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage file-snapshots,file-snapshots)
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage memory-snapshots -m 9,bounded-memory)
	$(call checkpoint-test,$<,,Euler checkpointing,-s 8 -r 2 --storage file-snapshots -m 9,bounded-file)

checkpoint-mesh-state-test-seq: checkpoint-mesh-state
	$(call checkpoint-test,$<,,Mesh checkpointing,-r 4 -c 2 --storage memory-snapshots -no-pv,memory-snapshots)
	$(call checkpoint-test,$<,,Mesh checkpointing,-r 4 -c 2 --storage file-snapshots -no-pv,file-snapshots)
	$(call checkpoint-test,$<,,Mesh checkpointing,-r 4 -c 2 --storage memory-snapshots -m 2 -no-pv,bounded-memory)
	$(call checkpoint-test,$<,,Mesh checkpointing,-r 4 -c 2 --storage file-snapshots -m 2 -no-pv,bounded-file)

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
