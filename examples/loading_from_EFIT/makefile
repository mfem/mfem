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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/loading_from_EFIT,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = mesh_maker mesh_loader B_field_projector_v0_0 B_field_projector_v0_1 B_field_projector_v1_1
PAR_EXAMPLES = 


ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif
SUBDIRS =
ifeq ($(MFEM_USE_AMGX),YES)
   SUBDIRS += amgx
endif
ifeq ($(MFEM_USE_GINKGO),YES)
   SUBDIRS += ginkgo
endif
ifeq ($(MFEM_USE_HIOP),YES)
   SUBDIRS += hiop
endif
ifeq ($(MFEM_USE_PETSC),YES)
   SUBDIRS += petsc
endif
ifeq ($(MFEM_USE_PUMI),YES)
   SUBDIRS += pumi
endif
ifeq ($(MFEM_USE_SUNDIALS),YES)
   SUBDIRS += sundials
endif
ifeq ($(MFEM_USE_SUPERLU),YES)
   SUBDIRS += superlu
endif
ifeq ($(MFEM_USE_MOONOLITH),YES)
   SUBDIRS += moonolith
endif
ifeq ($(MFEM_USE_CALIPER),YES)
   SUBDIRS += caliper
endif

SUBDIRS_ALL = $(addsuffix /all,$(SUBDIRS))
SUBDIRS_TEST = $(addsuffix /test,$(SUBDIRS))
SUBDIRS_CLEAN = $(addsuffix /clean,$(SUBDIRS))
SUBDIRS_TPRINT = $(addsuffix /test-print,$(SUBDIRS))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES) $(SUBDIRS_ALL)

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

# Additional dependencies
B_field_projector_v0_0 : B_field_vec_coeffs_v1.hpp
B_field_projector_v0_1 : B_field_vec_coeffs_v1.hpp
B_field_projector_v1_1 : B_field_vec_coeffs_v1.hpp

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
