# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
MFEM_BUILD_DIR ?= ../../
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/contact/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = obstacleProblem
EXAMPLES = $(SEQ_EXAMPLES)

ifeq ($(MFEM_USE_SUITESPARSE),NO)
$(EXAMPLES):
	$(error MFEM is not configured with SUITESPARSE)
endif

obstacleProblem: obstacleProblem.o problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) obstacleProblem.o problems.o IPsolver.o -o $@ $(MFEM_LIBS)

obstacleProblem.o: $(SRC)obstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

problems.o: $(SRC)problems.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

IPsolver.o: $(SRC)IPsolver.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

