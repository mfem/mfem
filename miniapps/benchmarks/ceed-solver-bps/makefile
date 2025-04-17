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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/benchmarks/ceed-solver-bps/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS =
PAR_MINIAPPS = solver-bp
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

EXTRA_SOURCES = preconditioners.cpp
EXTRA_HEADERS = kershaw.hpp rhs.hpp preconditioners.hpp
EXTRA_OBJECTS = $(EXTRA_SOURCES:.cpp=.o)

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rule for building solver-bp
solver-bp: $(SRC)solver-bp.cpp $(addprefix $(SRC),$(EXTRA_HEADERS)) \
 $(EXTRA_OBJECTS) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(EXTRA_OBJECTS) $(MFEM_LIBS)

# Rules for compiling *.o files
$(EXTRA_OBJECTS): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
solver-bp-test-par: solver-bp
	@$(call mfem-test,$<, $(RUN_MPI), CEED Solver BP,)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS) $(EXTRA_OBJECTS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@true
