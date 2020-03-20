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
OUT_DIR = output

DIRS = $(LIB_DIR) $(APP_DIR)

OBJ_FILES = $(BLD_DIR)/massmat.o $(BLD_DIR)/dofs.o $(BLD_DIR)/tools.o $(BLD_DIR)/fe_evol.o
APP_FILES = $(BLD_DIR)/advection.app $(BLD_DIR)/burgers.app $(BLD_DIR)/kpp.app $(BLD_DIR)/shallowwater.app $(BLD_DIR)/euler.app

MAIN_FILES = hypsys

ifeq ($(MFEM_USE_MPI), YES)
   MAIN_FILES += phypsys
   POBJ_FILES += $(BLD_DIR)/pdofs.o
   POBJ_FILES += $(BLD_DIR)/ptools.o
   POBJ_FILES += $(BLD_DIR)/pfe_evol.o
endif

# valgrind-test setup
PROBLEM = 4
VALGRIND-CONFIG = -tf 0.01 -dt 0.001 -p $(PROBLEM) -e 0

# grid-convergence-test setup
MESH0 = data/inline-3$(GEOM).mesh
MESH1 = data/inline-4$(GEOM).mesh
GEOM = quad

# ADVECTION
CONFIG = -p 0 -e 0 -o 1 -c 0 -tf 1 -dt 0.00001 -s 1 -vs 1000
CONFIG-NAME = p0e0o1-$(GEOM)

# SHALLOW WATER
MESH = data/periodic-tri.mesh
CONFIG = -p 3 -e 0 -o 3 -c 0 -tf 2.828427125 -dt 0.0001 -s 3 -vs 1000
CONFIG-NAME = p3e0o3-$(GEOM)

# EULER
MESH = data/periodic-square.mesh
CONFIG = -p 4 -e 0 -o 2 -c 0 -tf 2 -dt 0.00005 -s 3 -vs 1000
CONFIG-NAME = p4e0o2-$(GEOM)

## Makefile rules. #############################################################

# Keywords that are not associated with files (by default, all are).
.PHONY: all library hypsys phypsys clean valgrind-test grid-convergence-test sytle

# Delete the default suffixes.
.SUFFIXES:

# Define suffixes.
.SUFFIXES: .c .cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(OBJ_FILES) $(APP_FILES) $(MFEM_LIBS)

%.o: ../$(LIB_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

%.app: ../$(APP_DIR)/%.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $^ -o $@

all: $(MAIN_FILES)

library: $(OBJ_FILES) $(POBJ_FILES) $(APP_FILES)

hypsys: library
	$(MFEM_CXX) $(MFEM_FLAGS) hypsys.cpp -o $@ $(OBJ_FILES) $(APP_FILES) $(MFEM_LIBS)

phypsys: library
	$(MFEM_CXX) $(MFEM_FLAGS) phypsys.cpp -o $@ $(OBJ_FILES) $(POBJ_FILES) $(APP_FILES) $(MFEM_LIBS)

clean:
	@rm -f hypsys phypsys errors.txt *.gf *.mesh $(BLD_DIR)/*

valgrind-test:
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 1 -m data/inline-segment.mesh -r 4 -o 1
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 0 -m data/periodic-square.mesh -r 2 -o 2
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 0 -m data/star.mesh -r 1 -o 3
	valgrind ./hypsys $(VALGRIND-CONFIG) -c 1 -m data/wall-bdr-4hex.mesh -r 0 -o 2

# TODO
valgrind-parallel:
	# mpirun -np 2 valgrind --leak-check=yes ./par-hypsys -tf 0.1 -r 0
	# valgrind --gen-suppressions=yes phypsys $(VALGRIND-CONFIG)
	# valgrind --suppressions=$(MPI_HOME)/share/openmpi/openmpi-valgrind.supp phypsys $(VALGRIND-CONFIG)
	valgrind --suppressions=./my-mpi-suppressions.supp phypsys $(VALGRIND-CONFIG)

grid-convergence-advection:
	rm errors.txt; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH0) -r 4; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH1) -r 4; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH0) -r 5; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH1) -r 5; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH0) -r 6; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH1) -r 6; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH0) -r 7; \
	mv errors.txt $(OUT_DIR)/$(CONFIG-NAME).txt

grid-convergence-swe:
	rm errors.txt; \
	hypsys $(CONFIG) -m $(MESH) -r 2; \
	hypsys $(CONFIG) -m $(MESH) -r 3; \
	hypsys $(CONFIG) -m $(MESH) -r 4; \
	hypsys $(CONFIG) -m $(MESH) -r 5; \
	mv errors.txt $(OUT_DIR)/$(CONFIG-NAME).txt

grid-convergence-euler:
	rm errors.txt; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH) -r 2; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH) -r 3; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH) -r 4; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH) -r 5; \
	mpirun -np 8 ./phypsys $(CONFIG) -m $(MESH) -r 6; \
	mv errors.txt $(OUT_DIR)/$(CONFIG-NAME).txt

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
