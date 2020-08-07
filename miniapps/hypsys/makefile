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
EVL_DIR = fe_evol
LIB_DIR = lib
OUT_DIR = output

DIRS = $(APP_DIR) $(EVL_DIR) $(LIB_DIR)

LIB_FILES = $(BLD_DIR)/massmat.o $(BLD_DIR)/dofs.o $(BLD_DIR)/tools.o $(BLD_DIR)/bounds.o
EVL_FILES = $(BLD_DIR)/fe_evol.evl $(BLD_DIR)/galerkin.evl $(BLD_DIR)/monolithic_convex_limiting.evl
APP_FILES = $(BLD_DIR)/advection.app $(BLD_DIR)/burgers.app $(BLD_DIR)/kpp.app $(BLD_DIR)/buckley_leverett.app # scalar problems
APP_FILES+= $(BLD_DIR)/shallowwater.app $(BLD_DIR)/euler.app # hyperbolic systems

MAIN_FILES = hypsys

ifeq ($(MFEM_USE_MPI), YES)
   MAIN_FILES += phypsys
   PLIB_FILES += $(BLD_DIR)/pdofs.o
   PLIB_FILES += $(BLD_DIR)/ptools.o
	PLIB_FILES += $(BLD_DIR)/pbounds.o
   PEVL_FILES += $(BLD_DIR)/pgalerkin.evl
   PEVL_FILES += $(BLD_DIR)/pmonolithic_convex_limiting.evl
endif

# valgrind-test setup
PROBLEM = 0
VALGRIND-CONFIG = -tf 0.001 -dt 0.001 -p $(PROBLEM) -e 0

## Makefile rules. #############################################################

# Keywords that are not associated with files (by default, all are).
.PHONY: all library hypsys phypsys clean valgrind-test grid-convergence-test sytle

# Delete the default suffixes.
.SUFFIXES:

# Define suffixes.
.SUFFIXES: .c .cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(LIB_FILES) $(APP_FILES) $(MFEM_LIBS)

%.o: ../$(LIB_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

%.evl: ../$(EVL_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

%.app: ../$(APP_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

all: $(MAIN_FILES)

library: $(LIB_FILES) $(PLIB_FILES) $(EVL_FILES) $(PEVL_FILES) $(APP_FILES)

hypsys: library
	$(MFEM_CXX) $(MFEM_FLAGS) hypsys.cpp -o $@ $(LIB_FILES) $(EVL_FILES) $(APP_FILES) $(MFEM_LIBS)

phypsys: library
	$(MFEM_CXX) $(MFEM_FLAGS) phypsys.cpp -o $@ $(LIB_FILES) $(PLIB_FILES) $(EVL_FILES) $(PEVL_FILES) $(APP_FILES) $(MFEM_LIBS)

clean:
	@rm -f hypsys phypsys errors.txt *gf* grid* $(OUT_DIR)/* $(BLD_DIR)/*

valgrind-serial:
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 1 -m data/inline-4segment.mesh -r 4 -o 1
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 0 -m data/periodic-square.mesh -r 2 -o 2 -s 1
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 0 -m data/beam-quad.mesh -r 1 -o 3
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 1 -m data/wall-bdr-4hex.mesh -r 0 -o 2

# TODO does not work yet
valgrind-parallel:
	# mpirun -np 2 valgrind --leak-check=yes ./par-hypsys -tf 0.1 -r 0
	# valgrind --gen-suppressions=yes phypsys $(VALGRIND-CONFIG)
	# valgrind --suppressions=$(MPI_HOME)/share/openmpi/openmpi-valgrind.supp phypsys $(VALGRIND-CONFIG)
	valgrind --suppressions=./my-mpi-suppressions.supp phypsys $(VALGRIND-CONFIG)

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

ASTYLE = ~/astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS),"$(dir)/*.?pp")
FORMAT_FILES += hypsys.cpp
FORMAT_FILES += phypsys.cpp

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
