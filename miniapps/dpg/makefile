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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/dpg/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

DPG_REAL_SEQ_SRC = util/weakform.cpp util/blockstaticcond.cpp
DPG_REAL_PAR_SRC = $(DPG_REAL_SEQ_SRC) util/pweakform.cpp 
DPG_REAL_OBJ = $(DPG_REAL_PAR_SRC:.cpp=.o)

DPG_COMPLEX_SEQ_SRC = util/complexweakform.cpp util/complexstaticcond.cpp util/pml.cpp
DPG_COMPLEX_PAR_SRC = $(DPG_COMPLEX_SEQ_SRC) util/pcomplexweakform.cpp 
DPG_COMPLEX_OBJ = $(DPG_COMPLEX_PAR_SRC:.cpp=.o)

DIFFUSION_SRC = diffusion.cpp $(DPG_REAL_SEQ_SRC)
DIFFUSION_OBJ = $(DIFFUSION_SRC:.cpp=.o)

PDIFFUSION_SRC = pdiffusion.cpp $(DPG_REAL_PAR_SRC)
PDIFFUSION_OBJ = $(PDIFFUSION_SRC:.cpp=.o)

CONVECTIONDIFFUSION_SRC = convection-diffusion.cpp $(DPG_REAL_SEQ_SRC)
CONVECTIONDIFFUSION_OBJ = $(CONVECTIONDIFFUSION_SRC:.cpp=.o)

PCONVECTIONDIFFUSION_SRC = pconvection-diffusion.cpp $(DPG_REAL_PAR_SRC)
PCONVECTIONDIFFUSION_OBJ = $(PCONVECTIONDIFFUSION_SRC:.cpp=.o)

ACOUSTICS_SRC = acoustics.cpp $(DPG_COMPLEX_SEQ_SRC)
ACOUSTICS_OBJ = $(ACOUSTICS_SRC:.cpp=.o)

PACOUSTICS_SRC = pacoustics.cpp $(DPG_COMPLEX_PAR_SRC)
PACOUSTICS_OBJ = $(PACOUSTICS_SRC:.cpp=.o)

MAXWELL_SRC = maxwell.cpp $(DPG_COMPLEX_SEQ_SRC)
MAXWELL_OBJ = $(MAXWELL_SRC:.cpp=.o)

PMAXWELL_SRC = pmaxwell.cpp $(DPG_COMPLEX_PAR_SRC)
PMAXWELL_OBJ = $(PMAXWELL_SRC:.cpp=.o)

SEQ_MINIAPPS = diffusion convection-diffusion acoustics maxwell
PAR_MINIAPPS = pdiffusion pconvection-diffusion pacoustics pmaxwell

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

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

all: $(MINIAPPS)

diffusion: $(DIFFUSION_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(DIFFUSION_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

convection-diffusion: $(CONVECTIONDIFFUSION_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(CONVECTIONDIFFUSION_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pdiffusion: $(PDIFFUSION_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PDIFFUSION_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pconvection-diffusion: $(PCONVECTIONDIFFUSION_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PCONVECTIONDIFFUSION_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

acoustics: $(ACOUSTICS_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(ACOUSTICS_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

maxwell: $(MAXWELL_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(MAXWELL_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pacoustics: $(PACOUSTICS_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PACOUSTICS_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pmaxwell: $(PMAXWELL_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_OBJ) $(COMMON_LIB) $(MFEM_LIBS)


# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
diffusion-test-seq: diffusion
	@$(call mfem-test,$<,, DPG diffusion miniapp,\
	-m ../../data/star.mesh -o 3 -ref 1 -do 1 -prob 1 -sc)
convection-diffusion-test-seq: convection-diffusion
	@$(call mfem-test,$<,, DPG convection diffusion miniapp,\
	-m ../../data/star.mesh -o 2 -ref 1 -theta 0.0 -eps 1e-1 -beta '2 3')	
pdiffusion-test-par: pdiffusion
	@$(call mfem-test,$<, $(RUN_MPI), DPG pdiffusion miniapp,\
	-m ../../data/inline-quad.mesh -o 3 -sref 1 -pref 1 -theta 0.0 -prob 0)
pconvection-diffusion-test-par: pconvection-diffusion
	@$(call mfem-test,$<, $(RUN_MPI), DPG pconvection-diffusion miniapp,\
	-o 2 -ref 1 -prob 0 -eps 1e-1 -beta '4 2' -theta 0.0)	
acoustics-test-seq: acoustics
	@$(call mfem-test,$<,, DPG acoustics miniapp,\
	-ref 3 -o 1 -rnum 1.0)
maxwell-test-seq: maxwell
	@$(call mfem-test,$<,, DPG maxwell miniapp,\
	-m ../../data/inline-tri.mesh -ref 2 -o 1 -rnum 1.0)	
pacoustics-test-par: pacoustics
	@$(call mfem-test,$<, $(RUN_MPI), DPG pacoustics miniapp,\
	-o 3 -m ../../data/star.mesh -sref 1 -pref 0 -rnum 1.9 -sc -prob 0)
pmaxwell-test-par: pmaxwell
	@$(call mfem-test,$<, $(RUN_MPI), DPG pmaxwell miniapp,\
	-m ../../data/star.mesh -o 2 -sref 0 -pref 2 -rnum 0.5 -prob 0)	

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
	rm -f $(DPG_REAL_OBJ) $(DPG_COMPLEX_OBJ)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 
