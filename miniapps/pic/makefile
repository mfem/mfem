# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
MFEM_INSTALL_DIR ?= ../../mfem
SRC =
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

PAR_MINIAPPS =

ifeq ($(MFEM_USE_GSLIB),YES)
   PAR_MINIAPPS += electrostatic-pic
endif

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec pic-test-par
.PRECIOUS: %.o

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
ELECTROSTATIC_PIC_SRCS = electrostatic-pic.cpp ParticleMover.cpp FieldSolver.cpp
ELECTROSTATIC_PIC_OBJS = $(ELECTROSTATIC_PIC_SRCS:.cpp=.o)

electrostatic-pic: $(ELECTROSTATIC_PIC_OBJS) $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(ELECTROSTATIC_PIC_OBJS) $(COMMON_LIB) $(MFEM_LIBS)

electrostatic-pic.o: electrostatic-pic.cpp ParticleMover.hpp FieldSolver.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

ParticleMover.o: ParticleMover.cpp ParticleMover.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

FieldSolver.o: FieldSolver.cpp FieldSolver.hpp ParticleMover.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $<

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common


MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
pic-test-par: electrostatic-pic
	@$(call mfem-test,$<, mpirun -n 4, PIC miniapp,\
		-rdi 2 -npt 40960 -k 0.2855993321 -a 0.05 -nt 200 -nx 16 -ny 16 -O 1 -q 0.01181640625 -m 0.01181640625 -oci 1000 -dt 0.1)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf electrostatic-pic_* *.csv energy.csv
