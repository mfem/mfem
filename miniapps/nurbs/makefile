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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/nurbs/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = nurbs_ex1
PAR_MINIAPPS = nurbs_ex1p nurbs_ex11p
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp)
%-test-seq: %
	@$(call mfem-test,$<,, NURBS miniapp)

# Additional tests
EX1_ARGS_2 := -r 0 -o 4
EX1_ARGS_3 := -r 2
EX1_ARGS_4 := -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2
EX1_ARGS_5 := -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 0
EX1_ARGS_6 := -m ../..//data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 2
EX1_ARGS_7 := -m ../../data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 0
EX1_ARGS_8 := -m ../../data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 2
EX1_ARGS_9 := -m ../../data/ball-nurbs.mesh -o 2 --weak-bc -r 0
EX1_ARGS_10 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0
EX1_ARGS_11 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1

nurbs_ex1-test-seq: nurbs_ex1
	@$(call mfem-test,$<,, NURBS miniapp)
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_2))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_3))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_4))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_5))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_6))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_7))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_8))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_9))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_10))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_11))

EX1P_ARGS_1 := 
EX1P_ARGS_2 := -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp
EX1P_ARGS_3 := -m ../../data/ball-nurbs.mesh -o 2 --weak-bc -r 0
EX1P_ARGS_4 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0
EX1P_ARGS_5 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1

nurbs_ex1p-test-par: nurbs_ex1p
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_1))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_2))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_3))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_4))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_5))

EX11P_ARGS_1 := 

nurbs_ex11p-test-par: nurbs_ex11p
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX11P_ARGS_1))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh mesh.* sol.* mode_*
	@rm -rf Example1*
