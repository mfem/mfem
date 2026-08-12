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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/plasma/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = 
PAR_MINIAPPS = 
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

PLASMA_SUBDIRS = pic

SUBDIRS_ALL = $(addsuffix /all,$(PLASMA_SUBDIRS))
SUBDIRS_TEST = $(addsuffix /test,$(PLASMA_SUBDIRS))
SUBDIRS_TEST_NOCLEAN = $(addsuffix /test-noclean,$(PLASMA_SUBDIRS))
SUBDIRS_CLEAN = $(addsuffix /clean,$(PLASMA_SUBDIRS))
SUBDIRS_TPRINT = $(addsuffix /test-print,$(PLASMA_SUBDIRS))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec
.PRECIOUS: %.o

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS) $(SUBDIRS_ALL) 

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_TEST_NOCLEAN) \
   $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_TEST_NOCLEAN) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

# Rules for building the miniapps
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $< -o $@ $(COMMON_LIB) $(MFEM_LIBS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)
test: $(SUBDIRS_TEST)
test-noclean: $(SUBDIRS_TEST_NOCLEAN)
test-print: $(SUBDIRS_TPRINT)

# Testing: Specific execution options
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
