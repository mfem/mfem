# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/multidomain/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

MULTIDOMAIN_SRC = multidomain.cpp
MULTIDOMAIN_OBJ = $(MULTIDOMAIN_SRC:.cpp=.o)

SEQ_MINIAPPS =
PAR_MINIAPPS = multidomain multidomain_nd multidomain_rt
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# Rules to copy the *.mesh files - needed for running the sample runs when
# building out-of-source:
ifneq ($(SRC),)
MESH_FILES = $(notdir $(wildcard $(SRC)*.mesh))
$(MESH_FILES): %: $(SRC)%
	ln -sf $(<) .
multidomain: | $(MESH_FILES)
# The target 'copy-data' is used by the makefile in ../../tests/unit
.PHONY: copy-data
copy-data: | $(MESH_FILES)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
multidomain-test-par: multidomain
	@$(call mfem-test,$<, $(RUN_MPI), Multidomain miniapp,-tf 0.001)
multidomain_nd-test-par: multidomain_nd
	@$(call mfem-test,$<, $(RUN_MPI), Multidomain ND miniapp,-tf 0.001)
multidomain_rt-test-par: multidomain_rt
	@$(call mfem-test,$<, $(RUN_MPI), Multidomain RT iniapp,-tf 0.001)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf *_output
