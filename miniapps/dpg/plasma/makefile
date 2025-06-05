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
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
SRC = $(if $(MFEM_DIR:../../..=),$(MFEM_DIR)/miniapps/dpg/plasma,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

DPG_COMPLEX_SEQ_SRC = ../util/utils.cpp ../util/complexweakform.cpp \
	../util/complexstaticcond.cpp ../util/complexblockform.cpp
DPG_COMPLEX_PAR_SRC = $(DPG_COMPLEX_SEQ_SRC) ../util/pcomplexweakform.cpp ../util/pcomplexblockform.cpp ../util/blockcomplexhypremat.cpp  
DPG_COMPLEX_OBJ = $(DPG_COMPLEX_PAR_SRC:.cpp=.o)

LH_UTILS_SRC = utils/lh_utils.cpp ../util/maxwell_utils.cpp
LH_UTILS_OBJ = $(LH_UTILS_SRC:.cpp=.o)

LH_PRIMAL_SRC = lh-primal-dpg.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_PRIMAL_OBJ = $(LH_PRIMAL_SRC:.cpp=.o)

LH_ELD_FEM_SRC = lh-eld-fem.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_FEM_OBJ = $(LH_ELD_FEM_SRC:.cpp=.o)

LH_ELD_FEM_SYMM_SRC = lh-eld-fem-symm.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_FEM_SYMM_OBJ = $(LH_ELD_FEM_SYMM_SRC:.cpp=.o)

LH_ELD_DPG_SRC = lh-eld-dpg.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_DPG_OBJ = $(LH_ELD_DPG_SRC:.cpp=.o)

LH_ELD_FOSLS_DPG_SRC = lh-eld-fosls-dpg.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_FOSLS_DPG_OBJ = $(LH_ELD_FOSLS_DPG_SRC:.cpp=.o)

LH_ELD_DPG_FEM_SRC = lh-eld-dpg-fem.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)	
LH_ELD_DPG_FEM_OBJ = $(LH_ELD_DPG_FEM_SRC:.cpp=.o)

LH_ELD_WEAK_SRC = lh-eld-weak.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_WEAK_OBJ = $(LH_ELD_WEAK_SRC:.cpp=.o)

LH_ELD_DPG_FEM_WEAK_SRC = lh-eld-dpg-fem-weak.cpp $(DPG_COMPLEX_PAR_SRC) $(LH_UTILS_SRC)
LH_ELD_DPG_FEM_WEAK_OBJ = $(LH_ELD_DPG_FEM_WEAK_SRC:.cpp=.o)

SEQ_MINIAPPS = 
PAR_MINIAPPS = lh-primal-dpg lh-eld-fem lh-eld-fem-symm lh-eld-dpg  \
 lh-eld-dpg-fem lh-eld-weak lh-eld-dpg-fem-weak lh-eld-fosls-dpg

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

lh-primal-dpg: $(LH_PRIMAL_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_PRIMAL_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-fem: $(LH_ELD_FEM_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_FEM_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-fem-symm: $(LH_ELD_FEM_SYMM_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_FEM_SYMM_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-dpg: $(LH_ELD_DPG_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_DPG_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-fosls-dpg: $(LH_ELD_FOSLS_DPG_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_FOSLS_DPG_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-dpg-fem: $(LH_ELD_DPG_FEM_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_DPG_FEM_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-weak: $(LH_ELD_WEAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_WEAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

lh-eld-dpg-fem-weak: $(LH_ELD_DPG_FEM_WEAK_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ $(LH_ELD_DPG_FEM_WEAK_OBJ) $(COMMON_LIB) $(MFEM_LIBS)

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
	rm -f $(DPG_COMPLEX_OBJ) $(LH_UTILS_OBJ)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 
