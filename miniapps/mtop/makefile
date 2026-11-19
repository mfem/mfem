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

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mtop/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MTOP_COMMON_SRC = mtop_solvers.cpp

MTOP_COMMON_OBJ = $(MTOP_COMMON_SRC:.cpp=.o)

SEQ_MINIAPPS =
PAR_MINIAPPS = mtop_test_iso_elasticity
ifeq ($(MFEM_USE_LAPACK),YES)
   SEQ_MINIAPPS += projection_ghost_penalty_serial
   ifeq ($(MFEM_USE_MPI),YES)
      PAR_MINIAPPS += projection_ghost_penalty_parallel
   endif
endif
MINIAPPS = $(SEQ_MINIAPPS)
ifneq ($(MFEM_USE_MPI),NO)
   MINIAPPS += $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(MTOP_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

ifeq ($(MFEM_USE_LAPACK),YES)
projection_ghost_penalty_serial: projection_ghost_penalty_serial.o $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(MFEM_LIBS)
ifeq ($(MFEM_USE_MPI),YES)
projection_ghost_penalty_parallel: projection_ghost_penalty_parallel.o $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(MFEM_LIBS)
endif
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := MTop miniapp
ifeq ($(MFEM_USE_LAPACK),YES)
projection_ghost_penalty_serial-test-seq: projection_ghost_penalty_serial
	@$(call mfem-test,$<,,Projection ghost penalty,,YES)
ifeq ($(MFEM_USE_MPI),YES)
projection_ghost_penalty_parallel-test-par: projection_ghost_penalty_parallel
	@$(call mfem-test,$<,$(RUN_MPI),Parallel projection ghost penalty,,YES)
endif
endif

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
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf SeqHeat* ParHeat*
