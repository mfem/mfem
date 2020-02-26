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

# Sedov numerical tests
SEDOV_FILES = $(SRC)miniapps/test_sedov.cpp
SEDOV_TESTS = sedov_tests_cpu sedov_tests_debug
SEDOV_TESTS += $(if $(MFEM_CXX:nvcc=),,sedov_tests_cuda)
SEDOV_TESTS += $(if $(MFEM_CXX:nvcc=),,sedov_tests_cuda_uvm)
SEDOV_CPU_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.cpu.o)
SEDOV_DEBUG_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.debug.o)
SEDOV_CUDA_OBJ_FILES = $(if $(MFEM_CXX:nvcc=),,$(SEDOV_FILES:$(SRC)%.cpp=%.cuda.o))
SEDOV_CUDA_UVM_OBJ_FILES = $(if $(MFEM_CXX:nvcc=),,$(SEDOV_FILES:$(SRC)%.cpp=%.cuda_uvm.o))

SEQ_UNIT_TESTS = unit_tests $(SEDOV_TESTS)
PAR_UNIT_TESTS = $(SEDOV_TESTS)

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
	$(CCC) $(OBJECT_FILES) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

# Note: in this rule, we always use the full path to the source file as a
# workaround for an issue with coveralls.
$(OBJECT_FILES): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) -c $(abspath $(<)) $(MFEM_FLAGS) $(INCLUDES) -o $(@)

# Template rule for devices
# 1: postfix of 'unit_tests_', 2: prefix of '_OBJECT_FILES',
# 3: configuration string for the MFEM device, separated with '$(comma)'
comma=,
define sedov_tests
sedov_tests_$(1): SEDOV_TESTS_FLAGS=-DMFEM_SEDOV_TESTS -DMFEM_SEDOV_DEVICE='"$(3)"'
sedov_tests_$(1): unit_test_main.o $$(SEDOV_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(<) $$(SEDOV_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(SEDOV_$(2)_OBJ_FILES): %.$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) -c $$(abspath $$(<)) $$(MFEM_FLAGS) $$(INCLUDES) \
	   $$(SEDOV_TESTS_FLAGS) -o $$(@)
endef

$(eval $(call sedov_tests,cpu,CPU,cpu))
$(eval $(call sedov_tests,debug,DEBUG,debug))
$(eval $(call sedov_tests,cuda,CUDA,cuda))
$(eval $(call sedov_tests,cuda_uvm,CUDA_UVM,cuda:uvm))

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
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel unit tests,,SKIP-NO-VIS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f $(SEQ_UNIT_TESTS) $(PAR_UNIT_TESTS) *.o */*.o */*~ *~
	rm -rf *.dSYM output_meshes
