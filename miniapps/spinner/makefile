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
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/spinner/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SPINNER_SRC = spinner-lts.cpp
SPINNER_OBJ = $(BLOCK_SOLVERS_SRC:.cpp=.o)

SEQ_MINIAPPS = spinner-lts
PAR_MINIAPPS = 

ifeq ($(MFEM_USE_MPI),NO)
	MINIAPPS = $(SEQ_MINIAPPS)
else
	MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

SPINNER_INCLUDES = $(if $(SRC),-I$(MFEM_DIR))

%: %.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(MFEM_LIBS)

spinner: $(SPINNER_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(SPINNER_OBJ) $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS) $(SPINNER_OBJ)

clean-exec:
	@rm -rf mesh.* sol.* ParaView
