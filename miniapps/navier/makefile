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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/navier/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

NAVIER_COMMON_SRC = navier_solver.cpp
NAVIER_COMMON_OBJ = $(NAVIER_COMMON_SRC:.cpp=.o)

<<<<<<< HEAD
ifeq ($(MFEM_USE_GSLIB), YES)
	NAVIER_PARTICLE_COMMON_SRC = navier_particles.cpp
	NAVIER_PARTICLE_COMMON_OBJ = $(NAVIER_PARTICLE_COMMON_SRC:.cpp=.o)

   NAVIER_COMMON_SRC += NAVIER_PARTICLE_COMMON_SRC
   NAVIER_COMMON_OBJ += NAVIER_PARTICLE_COMMON_OBJ
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))

PAR_MINIAPPS = navier_mms navier_kovasznay navier_kovasznay_vs navier_tgv\
 navier_shear navier_3dfoc navier_turbchan
ifeq ($(MFEM_USE_GSLIB),YES)
   PAR_MINIAPPS += navier_cht navier_bifurcation
=======
PAR_MINIAPPS = navier_mms navier_kovasznay navier_kovasznay_vs navier_tgv\
 navier_shear navier_3dfoc navier_turbchan
ifeq ($(MFEM_USE_GSLIB),YES)
   PAR_MINIAPPS += navier_cht
>>>>>>> master
endif

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
<<<<<<< HEAD
.PHONY: all lib-common clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(if $(MFEM_USE_CUDA:YES=),$(CXX_XLINKER),$(CUDA_XLINKER))-rpath,$(abspath\
   $(MFEM_BUILD_DIR)/miniapps/common))
=======
.PHONY: all clean clean-build clean-exec
>>>>>>> master

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(NAVIER_COMMON_OBJ)
<<<<<<< HEAD
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS) $(COMMON_LIB)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
=======
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
>>>>>>> master
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

<<<<<<< HEAD
# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

=======
>>>>>>> master
PAR_TEST_EXECS = navier_mms navier_kovasznay navier_tgv
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

navier_mms-test-par: navier_mms
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_kovasznay-test-par: navier_kovasznay
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_tgv-test-par: navier_tgv
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
<<<<<<< HEAD
	@rm -rf tgv_out*.txt *_output ParaView Navier_Bifurcation_*
=======
	@rm -rf tgv_out*.txt *_output
>>>>>>> master
