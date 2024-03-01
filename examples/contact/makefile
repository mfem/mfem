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

SEQ_EXAMPLES = ObstacleProblem SphericalObstacleProblem DirichletObstacleProblem exQPContactBlockTL
PAR_EXAMPLES = ParObstacleProblem
EXAMPLES = $(SEQ_EXAMPLES) $(PAR_EXAMPLES)

ifeq ($(MFEM_USE_SUITESPARSE),NO)
$(SEQ_EXAMPLES):
	$(error MFEM is not configured with SUITESPARSE)
endif

ifeq ($(MFEM_USE_MUMPS),NO)
ifeq ($(MFEM_USE_MKL_CPARDISO), NO)
$(PAR_EXAMPLES):
	$(error MFEM is not configured with MUMPS or CPARDISO)
endif
endif

all: $(EXAMPLES)

ObstacleProblem: ObstacleProblem.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) ObstacleProblem.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)

SphericalObstacleProblem: SphericalObstacleProblem.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) SphericalObstacleProblem.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)

DirichletObstacleProblem: DirichletObstacleProblem.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) DirichletObstacleProblem.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)

DirichletObstacleProblemVariant: DirichletObstacleProblemVariant.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) DirichletObstacleProblemVariant.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)

TwoSidedDirichletObstacleProblem: TwoSidedDirichletObstacleProblem.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) TwoSidedDirichletObstacleProblem.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)


exQPContactBlockTL: exQPContactBlockTL.o Problems.o IPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) exQPContactBlockTL.o Problems.o IPsolver.o -o $@ $(MFEM_LIBS)

ParTest: ParTest.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) ParTest.o -o $@ $(MFEM_LIBS)


ObstacleProblem.o: $(SRC)ObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

SphericalObstacleProblem.o: $(SRC)SphericalObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

DirichletObstacleProblem.o: $(SRC)DirichletObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

DirichletObstacleProblemVariant.o: $(SRC)DirichletObstacleProblemVariant.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

TwoSidedDirichletObstacleProblem.o: $(SRC)TwoSidedDirichletObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

exQPContactBlockTL.o: $(SRC)exQPContactBlockTL.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

Problems.o: $(SRC)Problems.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

IPsolver.o: $(SRC)IPsolver.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

ParObstacleProblem: ParObstacleProblem.o ParProblems.o ParIPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) ParObstacleProblem.o ParProblems.o ParIPsolver.o -o $@ $(MFEM_LIBS)

ParSphericalObstacleProblem: ParSphericalObstacleProblem.o ParProblems.o ParIPsolver.o $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) ParSphericalObstacleProblem.o ParProblems.o ParIPsolver.o -o $@ $(MFEM_LIBS)

ParObstacleProblem.o: $(SRC)ParObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

ParSphericalObstacleProblem.o: $(SRC)ParSphericalObstacleProblem.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

ParProblems.o: $(SRC)ParProblems.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

ParIPsolver.o: $(SRC)ParIPsolver.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

ParTest.o: $(SRC)ParTest.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

clean: clean-build 

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

# For out-of-source builds, link the data files from the source tree:
ifneq ($(SRC),)
DATA_FILES = block1.mesh rotatedblock2.mesh
$(DATA_FILES): %: $(SRC)%
	ln -sf $(<) .
copy-data: | $(DATA_FILES)
# For out-of-source builds, the test and sample runs for 'field-interp' need
# data from the meshing miniapps directory:
exQPContactBlockTL: | mesh-data
.PHONY: mesh-data
mesh-data:
	$(MAKE) -C ./ copy-data
endif
