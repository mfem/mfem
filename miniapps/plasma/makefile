# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/plasma/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = 
PAR_MINIAPPS = stix1d stix2d stix3d braginskii transport2d dg_adv_diff
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_O=../common/fem_extras.o ../common/pfem_extras.o \
	../common/mesh_extras.o \
	cold_plasma_dielectric_solver.o cold_plasma_dielectric_coefs.o \
	plasma.o transport_solver.o braginskii_coefs.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
braginskii: braginskii.cpp braginskii_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ braginskii_solver.o $(COMMON_O) $(MFEM_LIBS)

#dg_adv_diff_test: dg_adv_diff_test.cpp transport_solver.hpp \
#	transport_solver.cpp transport_solver.o \
#	$(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
#	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ transport_solver.o $(COMMON_O) $(MFEM_LIBS)

#transport2d: transport2d.cpp transport_solver.hpp transport_solver.cpp \
#	transport_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
#	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ transport_solver.o $(COMMON_O) $(MFEM_LIBS)

%: $(SRC)%.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

#braginskii: braginskii.cpp braginskii_solver.o $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
#	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $@_solver.o $(COMMON_O) $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(COMMON_O) $(addsuffix _solver.o,$(MINIAPPS)): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

# Rules to copy the *.mesh files - needed for running the sample runs when
# building out-of-source:
ifneq ($(SRC),)
JOULE_MESH_FILES = cylinder-hex.mesh cylinder-tet.mesh
ifeq ($(MFEM_USE_NETCDF),YES)
   JOULE_MESH_FILES += cylinder-hex-q2.gen cylinder-tet-p2.gen
endif
TESLA_MESH_FILES = square-angled-pipe.mesh
VOLTA_MESH_FILES = llnl.mesh
$(JOULE_MESH_FILES) $(TESLA_MESH_FILES) $(VOLTA_MESH_FILES): %: $(SRC)%
	ln -sf $(<) .
joule: | $(JOULE_MESH_FILES)
tesla: | $(TESLA_MESH_FILES)
volta: | $(VOLTA_MESH_FILES)
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf STIX1D-AMR-Parallel* STIX2D-AMR-Parallel* STIX3D-AMR-Parallel*

