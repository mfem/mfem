# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/hdiv-linear-solver/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

HDIV_HEADERS = hdiv_linear_solver.hpp discrete_divergence.hpp change_basis.hpp \
					../solvers/lor_mms.hpp
HDIV_SRC = hdiv_linear_solver.cpp change_basis.cpp discrete_divergence.cpp
HDIV_OBJ = $(HDIV_SRC:.cpp=.o)

SEQ_MINIAPPS =
PAR_MINIAPPS = grad_div darcy

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

$(MINIAPPS):%: %.o $(HDIV_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(HDIV_OBJ) -o $@ $< $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
darcy-test-par: darcy
	@$(call mfem-test,$<, $(RUN_MPI), H(div) Darcy solver miniapp,\
	  -m ../../data/fichera-q2.mesh,SKIP-NO-VIS)
grad_div-test-par: grad_div
	@$(call mfem-test,$<, $(RUN_MPI), H(div) grad-div solver miniapp,-sp -ams \
	  -lor -hb -m ../../data/fichera-q2.mesh -rp 0,SKIP-NO-VIS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf mesh.* sol.* ParaView
