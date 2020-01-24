# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/tests/unit/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

CC = $(MFEM_CXX)
CCOPTS = -g
CCC = $(CC) $(CCOPTS)

# -I$(MFEM_DIR) is needed by some tests, e.g. to #include "general/text.hpp"
INCLUDES = -I$(or $(SRC:%/=%),.) -I$(MFEM_DIR)

SOURCE_FILES = $(SRC)unit_test_main.cpp $(sort $(wildcard $(SRC)*/*.cpp))
HEADER_FILES = $(SRC)catch.hpp
OBJECT_FILES = $(SOURCE_FILES:$(SRC)%.cpp=%.o)
DATA_DIR = data

DEV_UNIT_TESTS = unit_tests_cpu unit_tests_debug
DEV_UNIT_TESTS += $(if $(MFEM_CXX:nvcc=),,unit_tests_cuda)

SEQ_UNIT_TESTS = unit_tests $(DEV_UNIT_TESTS)
PAR_UNIT_TESTS = $(DEV_UNIT_TESTS)

DEV_SOURCE_FILES = miniapps/test_sedov.cpp
CPU_OBJECT_FILES = $(DEV_SOURCE_FILES:$(SRC)%.cpp=%.cpu.o)
DEBUG_OBJECT_FILES = $(DEV_SOURCE_FILES:$(SRC)%.cpp=%.debug.o)
CUDA_OBJECT_FILES = $(if $(MFEM_CXX:nvcc=),,$(DEV_SOURCE_FILES:$(SRC)%.cpp=%.cuda.o))

ifeq ($(MFEM_USE_MPI),NO)
   UNIT_TESTS = $(SEQ_UNIT_TESTS)
else
   UNIT_TESTS = $(PAR_UNIT_TESTS) $(SEQ_UNIT_TESTS)
endif

all: $(UNIT_TESTS)

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: all clean

unit_tests: $(OBJECT_FILES) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(OBJECT_FILES) $(INCLUDES) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

# Note: in this rule, we always use the full path to the source file as a
# workaround for an issue with coveralls.
$(OBJECT_FILES): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) -c $(abspath $(<)) $(INCLUDES) $(MFEM_FLAGS) -o $(@)

# Template rule for devices
# 1:target, 2:objects, 3:configuration string for the MFEM device, separated with $(comma)
comma=,
define unit_tests_dev
unit_tests_$(1): DEV_UNIT_FLAGS=-DMFEM_DEV_UNIT_TESTS -DMFEM_DEV_UNIT='"$(3)"'
unit_tests_$(1): $$(SRC)unit_test_main.o $$($(2)_OBJECT_FILES) $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(SRC)unit_test_main.o $$($(2)_OBJECT_FILES) $$(INCLUDES) $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$($(2)_OBJECT_FILES): %.$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) -c $$(abspath $$(<)) $$(INCLUDES) $$(MFEM_FLAGS) $$(DEV_UNIT_FLAGS) -o $$(@)
endef

$(eval $(call unit_tests_dev,cpu,CPU,cpu))
$(eval $(call unit_tests_dev,debug,DEBUG,debug))
$(eval $(call unit_tests_dev,cuda,CUDA,cuda))

$(DATA_DIR):
	ln -s $(SRC)$(DATA_DIR) .

# Testing
MFEM_TESTS = UNIT_TESTS
include $(MFEM_TEST_MK)

%-test-seq: %
	@$(call mfem-test,$<,, Unit tests,,SKIP-NO-VIS)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP)
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel unit tests,,SKIP-NO-VIS)
	@$(call mfem-test,$<, $(RUN_MPI) 2, Parallel unit tests,,SKIP-NO-VIS)
	@$(call mfem-test,$<, $(RUN_MPI) 3, Parallel unit tests,,SKIP-NO-VIS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f $(SEQ_UNIT_TESTS) $(PAR_UNIT_TESTS) *.o */*.o */*~ *~
	rm -rf *.dSYM output_meshes
