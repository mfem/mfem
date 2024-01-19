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
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
SRC = $(if $(MFEM_DIR:../../..=),$(MFEM_DIR)/miniapps/tribol/contact/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

CONTACT_SEQ_SRC = problems/problems.cpp problems/problems_util.cpp util/util.cpp ipsolver/IPsolver.cpp
CONTACT_SEC_OBJ = $(CONTACT_PAR_SRC:.cpp=.o)
CONTACT_PAR_SRC = $(CONTACT_SEQ_SRC) ipsolver/ParIPsolver.cpp problems/parproblems.cpp problems/parproblems_util.cpp util/mpicomm.cpp 
CONTACT_PAR_OBJ = $(CONTACT_PAR_SRC:.cpp=.o)

CONTACT_SRC = contact_driver.cpp $(CONTACT_SEQ_SRC)
CONTACT_OBJ = $(CONTACT_SRC:.cpp=.o)

PCONTACT_SRC = pcontact_driver.cpp $(CONTACT_PAR_SRC)
PCONTACT_OBJ = $(PCONTACT_SRC:.cpp=.o)

PCONTACT_TRIBOL_SRC = pcontact-tribol.cpp $(CONTACT_PAR_SRC)
PCONTACT_TRIBOL_OBJ = $(PCONTACT_TRIBOL_SRC:.cpp=.o)

PCONTACT_SINGLEMESH_SRC = pcontact_single_mesh.cpp $(CONTACT_PAR_SRC)
PCONTACT_SINGLEMESH_OBJ = $(PCONTACT_SINGLEMESH_SRC:.cpp=.o)

SEQ_MINIAPPS = contact_driver
PAR_MINIAPPS = pcontact_driver pcontact-tribol pcontact_single_mesh

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

%.o: $(SRC)%.cpp $(wildcard $(SRC)%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

util/%.o: $(SRC)util/%.cpp $(wildcard $(SRC)util/%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | lib-common
	mkdir -p $(@D)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

problems/%.o: $(SRC)problems/%.cpp $(wildcard $(SRC)problems/%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | lib-common
	mkdir -p $(@D)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@


all: $(MINIAPPS)

contact_driver: $(CONTACT_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(CONTACT_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pcontact_driver: $(PCONTACT_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PCONTACT_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pcontact-tribol: $(PCONTACT_TRIBOL_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PCONTACT_TRIBOL_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pcontact_single_mesh: $(PCONTACT_SINGLEMESH_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PCONTACT_SINGLEMESH_OBJ) $(COMMON_LIB) $(MFEM_LIBS)


# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
contact-test-seq: diffusion
	@$(call mfem-test,$<,, contact miniapp,)
pcontact-test-par: pcontact
	@$(call mfem-test,$<, $(RUN_MPI), pcontact miniapp,)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
	rm -f $(CONTACT_OBJ) $(PCONTACT_OBJ) $(PCONTACT_TRIBOL_OBJ) $(PCONTACT_SINGLEMESH_OBJ) 
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 
