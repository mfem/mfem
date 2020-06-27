# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../../..=),$(MFEM_DIR)/tests/convergence/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_TESTS = projection convergence complex 
PAR_TESTS = projectionp pcomplex diffusion
ifeq ($(MFEM_USE_MPI),NO)
   TESTS = $(SEQ_TESTS)
else
   TESTS = $(PAR_TESTS) $(SEQ_TESTS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean
.PRECIOUS: %.o

COMMON_O= conv_rates.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(TESTS)

# Rules for building the TESTS

%: $(SRC)%.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(COMMON_O) $($(TESTS)): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ $(SEQ_TESTS) $(PAR_TESTS)
	rm -rf *.dSYM *.TVD.*breakpoints


