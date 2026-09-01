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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/hdg/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = anisodiff anisodiff-hr convdiff mfem-logo navierstokes
PAR_MINIAPPS = panisodiff pconvdiff pnavierstokes
# navierstokes is driven by KINSOL -- it takes its globalisation from
# KIN_LINESEARCH rather than carrying a line search of its own -- so it needs
# SUNDIALS and drops out of the build without it. pnavierstokes carries the
# same #error guard and drops out for the same reason; note that an MPI build
# needs an MPI-enabled SUNDIALS, because defaults.mk adds -lsundials_nvecparallel
# and -lsundials_nvecmpiplusx whenever MFEM_USE_MPI=YES.
ifeq ($(MFEM_USE_SUNDIALS),NO)
   SEQ_MINIAPPS := $(filter-out navierstokes,$(SEQ_MINIAPPS))
   PAR_MINIAPPS := $(filter-out pnavierstokes,$(PAR_MINIAPPS))
endif
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
HDG_DEPS = darcyop

%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) $(HDG_DEPS).o
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $@.o $(HDG_DEPS).o $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(HDG_DEPS).o: $(SRC)$(HDG_DEPS).cpp $(SRC)$(HDG_DEPS).hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
anisodiff-test-seq: anisodiff
	@$(call mfem-test,$<,, HDG anisotropic diffusion miniapp,\
	-nx 20 -p 1 -o 3 -hb -dg -amr 2)
convdiff-test-seq: convdiff
	@$(call mfem-test,$<,, HDG convection-diffusion miniapp,\
	-m ../../data/amr-quad.mesh -p 2 -o 3 -hb -dg -up -bcn -r 2)
# -rtol 1e-12 and -chk are both load-bearing. Plane Poiseuille at order >= 2
# has a degree-2 exact solution that lies in the discrete space, so the errors
# come back at round-off -- that is this miniapp's sharpest property, and
# mfem-test grades on the exit code alone, so without -chk nothing checks it.
# The default -rtol of 1e-6 stops the Newton far above the discretisation and
# would fail the check for that reason rather than a real one.
navierstokes-test-seq: navierstokes
	@$(call mfem-test,$<,, HDG Navier-Stokes miniapp,\
	-p 1 -nx 8 -ny 8 -o 2 -hb -rtol 1e-12 -chk 1e-11)
mfem-logo-test-seq: mfem-logo
	@$(call mfem-test,$<,, HDG MFEM logo miniapp,\
	-nx 50 -o 3 -hb -brt -a 1e3 -ks 1e-3 -c 2e4)
panisodiff-test-par: panisodiff
	@$(call mfem-test,$<, $(RUN_MPI), Parallel HDG anisotropic diffusion miniapp,\
	-nx 20 -p 1 -o 3 -hb -dg -amr 2)
pconvdiff-test-par: pconvdiff
	@$(call mfem-test,$<, $(RUN_MPI), Parallel HDG convection-diffusion miniapp,\
	-m ../../data/amr-quad.mesh -p 2 -o 3 -hb -dg -up -bcn -rs 1 -rp 1)
# -rtol and -chk are load-bearing here for the same reason as the serial
# target above: mfem-test grades on the exit code alone, so without -chk this
# checks only that the binary ran.
pnavierstokes-test-par: pnavierstokes
	@$(call mfem-test,$<, $(RUN_MPI), Parallel HDG Navier-Stokes miniapp,\
	-p 1 -nx 8 -ny 8 -o 2 -hb -rtol 1e-12 -chk 1e-11)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f anisodiff*.mesh sol_t*.gf sol_q*.gf mfem-logo.mesh convdiff*.mesh
	@rm -f navierstokes*.mesh
	@rm -f panisodiff*.mesh pconvdiff*.mesh pnavierstokes*.mesh
	@rm -rf Anisodiff* Convdiff* Mfem-logo* PAnisodiff* PConvdiff*
