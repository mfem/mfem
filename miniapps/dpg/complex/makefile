# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(MFEM_DIR)/miniapps/dpg/complex
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SRC_DIRS := $(SRC)/util

INC_DIRS := $(shell find $(SRC_DIRS) ./ -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

DPG_LIB_SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
DPG_LIB_OBJS := $(DPG_LIB_SRCS:%=%.o)
DPG_LIB_DEPS := $(DPG_LIB_OBJS:.o=.d)
DPG_LIB := libdpg_complex.a

EXAMPLE_SRCS := $(shell find $(SRC) -name '*.cpp')
EXAMPLE_OBJS := $(EXAMPLE_SRCS:%=%.o)
EXAMPLE_DEPS := $(EXAMPLE_OBJS:.o=.d)

EXAMPLES = acoustics
PAR_EXAMPLES = pacoustics

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(EXAMPLES)
else
   MINIAPPS = $(PAR_EXAMPLES) $(EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

all: $(MINIAPPS)
	@echo $(MINIAPPS)

acoustics: $(SRC)/acoustics.cpp.o $(DPG_LIB)
	$(MFEM_CXX) -o $@ $^ $(MFEM_LIBS)

pacoustics: $(SRC)/pacoustics.cpp.o $(DPG_LIB)
	$(MFEM_CXX) -o $@ $^ $(MFEM_LIBS)

%.cpp.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(INC_FLAGS) -MMD -MP -c $< -o $@

$(DPG_LIB): $(DPG_LIB_OBJS)
	ar ru $@ $^
	ranlib $@

PAR_TEST_EXECS = 
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	@echo $(MINIAPPS)
	rm -f $(DPG_LIB) $(DPG_LIB_OBJS) $(DPG_LIB_DEPS) $(EXAMPLE_OBJS) $(EXAMPLE_DEPS) $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

# clean-exec:
# 	@rm -rf tgv_out*.txt *_output

-include $(DPG_LIB_DEPS)
-include $(EXAMPLE_DEPS)