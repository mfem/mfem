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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/hooke/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

HOOKE_SRC = hooke.cpp\
	operators/elasticity_operator.cpp\
	operators/elasticity_gradient_operator.cpp\
	preconditioners/diagonal_preconditioner.cpp
HOOKE_OBJ = $(HOOKE_SRC:.cpp=.o)

SEQ_MINIAPPS =
PAR_MINIAPPS = hooke
# The hooke miniapp crashes the AMD HIP compiler, so for now we disable it
# when HIP is enabled:
ifeq ($(MFEM_USE_HIP),YES)
   PAR_MINIAPPS =
endif
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif
MFEM_TESTS = MINIAPPS

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

HOOKE_INCLUDES = $(if $(SRC),-I$(MFEM_DIR))

hooke: $(HOOKE_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(HOOKE_OBJ) $(MFEM_LIBS)

$(HOOKE_OBJ): %.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(MFEM_CXX) $(MFEM_FLAGS) $(HOOKE_INCLUDES) -c $< -o $@

include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
hooke-test-par: hooke
	@$(call mfem-test,$<, $(RUN_MPI), Hooke miniapp,-rs 0)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS) $(HOOKE_OBJ)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf mesh.* sol.* ParaView
