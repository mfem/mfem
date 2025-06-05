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
MFEM_DIR ?= ../../../..
MFEM_BUILD_DIR ?= ../../../..
SRC = $(if $(MFEM_DIR:../../../..=),$(MFEM_DIR)/miniapps/dpg/plasma/other,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

DPG_COMPLEX_SEQ_SRC = ../../util/utils.cpp ../../util/complexweakform.cpp \
	../../util/complexstaticcond.cpp ../../util/complexblockform.cpp
DPG_COMPLEX_PAR_SRC = $(DPG_COMPLEX_SEQ_SRC) ../../util/pcomplexweakform.cpp ../../util/pcomplexblockform.cpp ../../util/blockcomplexhypremat.cpp  
DPG_COMPLEX_OBJ = $(DPG_COMPLEX_PAR_SRC:.cpp=.o)

MAXWELL_SRC = ../../util/maxwell_utils.cpp
MAXWELL_OBJ = $(MAXWELL_SRC:.cpp=.o)

PMAXWELL_TOKAMAK_SRC = pmaxwell-tokamak.cpp $(DPG_COMPLEX_PAR_SRC)
PMAXWELL_TOKAMAK_OBJ = $(PMAXWELL_TOKAMAK_SRC:.cpp=.o)

PMAXWELL_FEM_TOKAMAK_SRC = pmaxwell-fem-tokamak.cpp $(DPG_COMPLEX_PAR_SRC)
PMAXWELL_FEM_TOKAMAK_OBJ = $(PMAXWELL_FEM_TOKAMAK_SRC:.cpp=.o)

PMAXWELL_PRIMAL_TOKAMAK_SRC = pmaxwell-primal-tokamak.cpp $(DPG_COMPLEX_PAR_SRC)
PMAXWELL_PRIMAL_TOKAMAK_OBJ = $(PMAXWELL_PRIMAL_TOKAMAK_SRC:.cpp=.o)

PMAXWELL_UW_TOKAMAK_SRC = pmaxwell-uw-tokamak.cpp $(DPG_COMPLEX_PAR_SRC)
PMAXWELL_UW_TOKAMAK_OBJ = $(PMAXWELL_UW_TOKAMAK_SRC:.cpp=.o)

PMAXWELL_UW_LH_SRC = pmaxwell-uw-lh.cpp $(DPG_COMPLEX_PAR_SRC) $(MAXWELL_SRC)
PMAXWELL_UW_LH_OBJ = $(PMAXWELL_UW_LH_SRC:.cpp=.o)

PMAXWELL_FEM_LH_SRC = pmaxwell-fem-lh.cpp $(DPG_COMPLEX_PAR_SRC) $(MAXWELL_SRC)
PMAXWELL_FEM_LH_OBJ = $(PMAXWELL_FEM_LH_SRC:.cpp=.o)

SEQ_MINIAPPS = 
PAR_MINIAPPS = pmaxwell-tokamak pmaxwell-fem-tokamak \
 pmaxwell-primal-tokamak pmaxwell-uw-tokamak  \
 pmaxwell-uw-lh pmaxwell-fem-lh 

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

all: $(MINIAPPS)

pmaxwell-tokamak: $(PMAXWELL_TOKAMAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_TOKAMAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

pmaxwell-fem-tokamak: $(PMAXWELL_FEM_TOKAMAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_FEM_TOKAMAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)	

pmaxwell-primal-tokamak: $(PMAXWELL_PRIMAL_TOKAMAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_PRIMAL_TOKAMAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)	

pmaxwell-uw-tokamak: $(PMAXWELL_UW_TOKAMAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_UW_TOKAMAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)		

pmaxwell-uw-lh: $(PMAXWELL_UW_LH_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_UW_LH_OBJ) $(COMMON_LIB) $(MFEM_LIBS)		

pmaxwell-fem-lh: $(PMAXWELL_FEM_LH_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(PMAXWELL_FEM_LH_OBJ) $(COMMON_LIB) $(MFEM_LIBS)		

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
	rm -f $(DPG_COMPLEX_OBJ) $(MAXWELL_OBJ)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 
