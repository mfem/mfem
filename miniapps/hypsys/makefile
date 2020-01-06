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
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Code directories in logical order
APP_DIR = apps
BLD_DIR = build
LIB_DIR = lib

DIRS = $(LIB_DIR) $(APP_DIR)

OBJ_FILES = $(BLD_DIR)/massmat.o
APP_FILES = $(BLD_DIR)/advection.app

SERIAL = hypsys
PARALLEL = hypsys par-hypsys

ifeq ($(MFEM_USE_MPI),NO)
   EXEC_FILES = $(SERIAL)
else
   EXEC_FILES = $(PARALLEL)
endif

## Makefile rules. #############################################################

# Keywords that are not associated with files (by default, all are).
.PHONY: all library exec clean sytle

# Delete the default suffixes.
.SUFFIXES:

# Define suffixes.
.SUFFIXES: .c .cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(BLD_DIR)/* $(MFEM_LIBS)

%.o: ../$(LIB_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

%.app: ../$(APP_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

all: library exec

library: $(OBJ_FILES) $(APP_FILES)

exec: $(EXEC_FILES)

clean:
	@rm -f hypsys par-hypsys errors.txt *.gf *.mesh
	@rm -f $(BLD_DIR)/*

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS),"$(dir)/*.?pp")
FORMAT_FILES += hypsys.cpp
FORMAT_FILES += par-hypsys.cpp

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
