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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/optimization/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

OPTPROBLEM_TEST_SRC = optproblem_test.cpp optproblem.cpp functionals.cpp
OPTPROBLEM_TEST_OBJ = $(OPTPROBLEM_TEST_SRC:.cpp=.o)

# The test runs in serial, or in parallel when MFEM is built with MPI
ifeq ($(MFEM_USE_MPI),NO)
   SEQ_MINIAPPS = optproblem_test
   PAR_MINIAPPS =
   MINIAPPS = $(SEQ_MINIAPPS)
else
   SEQ_MINIAPPS =
   PAR_MINIAPPS = optproblem_test
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

%.o: $(SRC)%.cpp $(SRC)optproblem.hpp $(SRC)functionals.hpp $(MFEM_LIB_FILE)\
 $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

optproblem_test: $(OPTPROBLEM_TEST_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(OPTPROBLEM_TEST_OBJ) $(MFEM_LIBS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs. The test has no -vis option, so the fifth
# mfem-test argument suppresses the default -no-vis.
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TEST_NAME := Optimization miniapp
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TEST_NAME),,SKIP-NO-VIS)
%-test-seq: %
	@$(call mfem-test,$<,, $(TEST_NAME),,SKIP-NO-VIS)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ optproblem_test
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
