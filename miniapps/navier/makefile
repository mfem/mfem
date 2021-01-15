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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/navier/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

NAVIER_COMMON_SRC = navier_solver.cpp ortho_solver.cpp
NAVIER_COMMON_OBJ = $(NAVIER_COMMON_SRC:.cpp=.o)

PAR_MINIAPPS = navier_mms navier_kovasznay navier_kovasznay_vs navier_tgv navier_shear navier_3dfoc

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(NAVIER_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

PAR_TEST_EXECS = navier_mms navier_kovasznay navier_tgv
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

navier_mms-test-par: navier_mms
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_kovasznay-test-par: navier_kovasznay
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_tgv-test-par: navier_tgv
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf tgv_out*.txt *_output
