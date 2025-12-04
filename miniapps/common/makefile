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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/common/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Default target
all: lib-common

# Include defaults.mk to get the definition of BUILD_SOFLAGS
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

USER_CONFIG = $(MFEM_BUILD_DIR)/config/user.mk
-include $(USER_CONFIG)

MFEM_LIB_FILE = mfem_is_not_built
ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)

   XLINKER = $(MFEM_XLINKER)

   BUILD_REAL_DIR = $(realpath .)
   BUILD_SOFLAGS := $(subst libmfem.,libmfem-common.,$(BUILD_SOFLAGS))

   # Internal shortcuts
   override static = $(if $(MFEM_STATIC:YES=),,YES)
   override shared = $(if $(MFEM_SHARED:YES=),,YES)
endif

SEQ_MINIOBJS = mesh_extras.o fem_extras.o
ifeq ($(MFEM_USE_MPI),NO)
   MINIOBJS = $(SEQ_MINIOBJS)
else
   MINIOBJS = $(SEQ_MINIOBJS) pfem_extras.o dist_solver.o
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean

# Remove built-in rule
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%.o: $(SRC)%.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_PICFLAG) $(MFEM_FLAGS) -c $(<) -o $(@)

lib-common: $(if $(static),libmfem-common.a)\
	  $(if $(shared),libmfem-common.$(SO_EXT))

libmfem-common.a: $(MINIOBJS)
	$(AR) $(ARFLAGS) $(@) $(MINIOBJS)
	$(RANLIB) $(@)

libmfem-common.$(SO_VER): $(MINIOBJS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(BUILD_SOFLAGS) $(MINIOBJS) \
	  $(EXT_LIBS) -o $(@)

libmfem-common.$(SO_EXT): libmfem-common.$(SO_VER)
	ln -sf $(<) $(@)

# Generate an error message if the MFEM library is not configured and exit
$(CONFIG_MK):
	$(error The MFEM library is not configured)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)
clean:
	rm -rf *.o *~ libmfem-common.*
	rm -rf *.dSYM *.TVD.*breakpoints
