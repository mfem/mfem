# Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/ufl/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex1 bp3
PAR_EXAMPLES =

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

SUBDIRS =
SUBDIRS_ALL = $(addsuffix /all,$(SUBDIRS))
SUBDIRS_TEST = $(addsuffix /test,$(SUBDIRS))
SUBDIRS_CLEAN = $(addsuffix /clean,$(SUBDIRS))
SUBDIRS_TPRINT = $(addsuffix /test-print,$(SUBDIRS))

.SUFFIXES:
.SUFFIXES: .ufl
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.ufl

# Replace the default implicit rule for *.ufl files
XFL_FLAGS = -x c++ $(strip $(MFEM_FLAGS)) -Wno-unused-variable
XFL_INC = -I $(MFEM_DIR)/general -I $(MFEM_DIR)
XFL_LIB = $(MFEM_LIBS)
%: $(SRC)%.ufl $(MFEM_LIB_FILE) $(CONFIG_MK)
	../../xfl $@.ufl | $(MFEM_CXX) $(XFL_FLAGS) -o $@ $(XFL_INC) $(XFL_LIB) -

all: $(EXAMPLES) $(SUBDIRS_ALL)

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh
	@rm -f sol.*
