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
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/navier/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SRC_DIRS := ./lib ./kernels ./rans

INC_DIRS := $(shell find $(SRC_DIRS) ./ -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

NAVIER_LIB_SRCS := $(shell find $(SRC_DIRS) -name '*.cpp')
NAVIER_LIB_OBJS := $(NAVIER_LIB_SRCS:%=%.o)
NAVIER_LIB_DEPS := $(NAVIER_LIB_OBJS:.o=.d)
NAVIER_LIB := libnavier.a

EXAMPLE_SRCS := $(shell find ./examples -name '*.cpp')
EXAMPLE_OBJS := $(EXAMPLE_SRCS:%=%.o)
EXAMPLE_DEPS := $(EXAMPLE_OBJS:.o=.d)

TEST_SRCS := $(shell find ./tests -name '*.cpp')
TEST_OBJS := $(TEST_SRCS:%=%.o)
TEST_DEPS := $(TEST_OBJS:.o=.d)

PAR_MINIAPPS = navier_mms navier_kovasznay navier_kovasznay_vs navier_tgv\
 navier_shear navier_3dfoc navier_turbchan navier_tgv2d
ifeq ($(MFEM_USE_GSLIB),YES)
   PAR_MINIAPPS += navier_cht
endif

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

test_%: ./tests/test_%.cpp.o $(NAVIER_LIB)
	$(MFEM_CXX) -o $@ $^ $(MFEM_LIBS)

navier_%: ./examples/navier_%.cpp.o $(NAVIER_LIB)
	$(MFEM_CXX) -o $@ $^ $(MFEM_LIBS)

%.cpp.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) $(INC_FLAGS) -MMD -MP -c $< -o $@

$(NAVIER_LIB): $(NAVIER_LIB_OBJS)
	ar ru $@ $^
	ranlib $@

all: $(MINIAPPS)

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
	rm -f $(NAVIER_LIB) $(NAVIER_LIB_OBJS) $(NAVIER_LIB_DEPS) $(EXAMPLE_OBJS) $(EXAMPLE_DEPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf tgv_out*.txt *_output

-include $(NAVIER_LIB_DEPS)
-include $(EXAMPLE_DEPS)