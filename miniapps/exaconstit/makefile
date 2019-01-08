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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/exaconstit/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

EXECUTABLES = mechanics_driver

CPP_FILES = $(wildcard $(SRC)*.cpp)
OBJ_FILES = $(CPP_FILES:.cpp=.o)
COMMON_OBJ = $(filter-out $(foreach EXE,$(EXECUTABLES),$(EXE).o),$(OBJ_FILES))	
LINKED_LIB = -L ./ -luserumat -L/usr/tce/packages/gcc/gcc-7.3.1/lib64/ -lgfortran

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

all: $(EXECUTABLES)

# Rules for building the executable
$(EXECUTABLES): \
%: $(MFEM_LIB_FILE) $(CONFIG_MK) $(OBJ_FILES) 
	$(MFEM_CXX) $(MFEM_FLAGS) $(COMMON_OBJ) $(@).o -o $@ $(MFEM_LIBS) $(LINKED_LIB)

# Rules for compiling the object files
$(OBJ_FILES): \
%.o: $(SRC)%.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(EXECUTABLES)
	rm -rf *.dSYM *.TVD.*breakpoints *core

clean-exec:
	rm -f pressure* deformation* mesh.* 
	rm -rf refined* sol*
