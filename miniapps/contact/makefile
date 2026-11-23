# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/contact/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

# Include defaults.mk to get XLINKER
DEFAULTS_MK = $(MFEM_DIR)/config/defaults.mk
include $(DEFAULTS_MK)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Resolve the workspace alias recorded in this installation's dependency paths.
CONTACT_WORKSPACE_ROOT := $(abspath $(MFEM_DIR)/../..)
MFEM_FLAGS := $(subst /usr/WS1/socratis/nnl,$(CONTACT_WORKSPACE_ROOT),$(MFEM_FLAGS))
MFEM_LIBS := $(subst /usr/WS1/socratis/nnl,$(CONTACT_WORKSPACE_ROOT),$(MFEM_LIBS))

# Device-enabled Tribol dependencies require C++20; MFEM remains C++17.
ifeq ($(MFEM_USE_TRIBOL),YES)
ifneq ($(filter YES,$(MFEM_USE_CUDA) $(MFEM_USE_HIP)),)
   MFEM_FLAGS := $(filter-out -std=c++17,$(MFEM_FLAGS)) -std=c++20
endif
endif


ifeq ($(MFEM_USE_TRIBOL)$(MFEM_USE_MPI),YESYES)
   MINIAPPS = contact contact_examples
   CONTACT_SRC = elastoperator.cpp optcontactproblem.cpp ip.cpp solver_utils.cpp neo_hookean.cpp contact_driver.cpp driver_output.cpp mesh_generator.cpp
else
   MINIAPPS = 
   CONTACT_SRC = 
   .PHONY: contact contact_examples
   contact contact_examples:
	@echo "Rebuild MFEM with MFEM_USE_TRIBOL=YES and MFEM_USE_MPI=YES to build the contact miniapp."
endif

CONTACT_OBJ = $(CONTACT_SRC:.cpp=.o)
.DEFAULT_GOAL := all
ifeq ($(strip $(MINIAPPS)),)
all: contact contact_examples
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp
%.o: %.cpp

ifneq ($(strip $(MINIAPPS)),)
contact contact_examples: %: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONTACT_OBJ) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@.o
	$(MFEM_CXX) $(filter-out -x=cu,$(MFEM_FLAGS)) $@.o -o $@ $(CONTACT_OBJ) $(MFEM_LIBS) \
		-l$(patsubst lib%,%,$(basename $(notdir $(MFEM_LIB_FILE))))

endif

%.o: $(SRC)%.cpp $(wildcard $(SRC)%.hpp) $(MFEM_LIB_FILE)\
 $(CONFIG_MK) | 
	@mkdir -p $(dir $@)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

ip.o driver_output.o: $(SRC)contact_output.hpp
contact_driver.o driver_output.o: $(SRC)driver_output.hpp
driver_output.o: $(SRC)ip.hpp $(SRC)optcontactproblem.hpp $(SRC)elastoperator.hpp
contact_driver.o: $(SRC)solver_utils.hpp
contact contact_examples contact_driver.o: $(SRC)contact_driver.hpp
contact contact_examples mesh_generator.o: $(SRC)mesh_generator.hpp
elastoperator.o: $(SRC)neo_hookean.hpp
ip.o: $(SRC)optcontactproblem.hpp $(SRC)elastoperator.hpp $(SRC)solver_utils.hpp
optcontactproblem.o: $(SRC)elastoperator.hpp
contact contact_examples contact_driver.o: $(SRC)elastoperator.hpp $(SRC)optcontactproblem.hpp $(SRC)ip.hpp

clean: clean-build

clean-build:
	rm -f $(CONTACT_OBJ) $(MINIAPPS:=.o) *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView 

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)
