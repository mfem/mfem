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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/thermal/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS =
PAR_MINIAPPS = fourier fourier_nl fourier_flux fourier_nl_flux \
	fourier_refine fourier_flux_refine ex1p_nl
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_O=../common/pfem_extras.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
%: $(SRC)%.cpp %_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $@_solver.o $(COMMON_O) $(MFEM_LIBS)

fourier_refine: fourier_refine.cpp fourier_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ fourier_solver.o $(COMMON_O) $(MFEM_LIBS)

fourier_flux_refine: fourier_flux_refine.cpp fourier_flux_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ fourier_flux_solver.o $(COMMON_O) $(MFEM_LIBS)

curve_mesh: curve_mesh.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

ncd2mesh: ncd2mesh.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

ex1p_nl: ex1p_nl.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(COMMON_O) $(addsuffix _solver.o,$(MINIAPPS)): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
fourier-test-par: fourier
	@$(call mfem-test,$<, $(RUN_MPI), Thermal miniapp,\
	)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf Fourier_*
