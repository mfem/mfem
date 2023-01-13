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
#

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mhd/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
OBJ = InitialConditions.o
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

PAR_EXAMPLES = rMHDp
EXAMPLES = $(PAR_EXAMPLES)

# Remove built-in rule
%: %.cpp
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(OBJ) $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

rMHDp: rMHDp.cpp $(OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@  $(OBJ) $(MFEM_LIBS)

test: test.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@  $(MFEM_LIBS)

all: $(EXAMPLES) 

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec imMHDp
.PRECIOUS: %.o

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf rMHDp

clean-data:
	rm -f *.00* *.sol
