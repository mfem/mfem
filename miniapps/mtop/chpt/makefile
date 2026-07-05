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
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
MFEM_INSTALL_DIR ?= ../../../mfem
SRC = $(if $(MFEM_DIR:../../..=),$(MFEM_DIR)/miniapps/mtop/chpt/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = mtop_ex_scalar_chpt mtop_ex_adapt_chpt mtop_ex_vector_chpt \
   mtop_ex_fixedslot mtop_ex_file mtop_ex_revolve mtop_ex_dyn \
   mtop_ex_rev_fixedslot
PAR_MINIAPPS =
MINIAPPS_CLEAN := $(SEQ_MINIAPPS) $(PAR_MINIAPPS)

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
mtop_ex_scalar_chpt: $(SRC)ex_scalar_field.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_adapt_chpt: $(SRC)ex_scalar_adaptive.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_vector_chpt: $(SRC)ex_vector_field.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_fixedslot: $(SRC)ex_fixedslot_scalar_vector.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_file: $(SRC)ex_fileio_scalar_vector_mfem.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_revolve: $(SRC)ex_revolve.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_dyn: $(SRC)ex_dynamic_chpt.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

mtop_ex_rev_fixedslot: $(SRC)ex_revolve_chpt_fixedslot.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

TEST_NAME := MTop checkpointing miniapp
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(MINIAPPS_CLEAN)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	rm -rf chk_scalar_files chk_vector_files dyn_ckpts
	rm -f dynamic_ckpts.bin revolve_ckpts.bin
