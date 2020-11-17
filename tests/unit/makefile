# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

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

SOURCE_FILES := $(sort $(wildcard $(SRC)*/*.cpp))
CEED_SOURCE_FILES = $(sort $(wildcard $(SRC)ceed/*.cpp))
SOURCE_FILES := $(filter-out $(CEED_SOURCE_FILES), $(SOURCE_FILES))
HEADER_FILES = $(SRC)catch.hpp $(SRC)unit_tests.hpp
OBJECT_FILES = $(SOURCE_FILES:$(SRC)%.cpp=%.o)
DATA_DIR = data

SEQ_MAIN_OBJ = unit_test_main.o
PAR_MAIN_OBJ = punit_test_main.o
CUDA_MAIN_OBJ = cunit_test_main.o

# Sedov numerical seq/par files and tests
SEDOV_FILES = $(SRC)miniapps/test_sedov.cpp

USE_CUDA := $(MFEM_USE_CUDA:NO=)
SEQ_SEDOV_TESTS = sedov_tests_cpu sedov_tests_debug
SEQ_SEDOV_TESTS += $(if $(USE_CUDA),sedov_tests_cuda)
SEQ_SEDOV_TESTS += $(if $(USE_CUDA),sedov_tests_cuda_uvm)
PAR_SEDOV_TESTS = $(SEQ_SEDOV_TESTS:%=p%)

SEQ_SEDOV_CPU_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.cpu.o)
SEQ_SEDOV_DEBUG_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.debug.o)
SEQ_SEDOV_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.cuda.o))
SEQ_SEDOV_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.cuda_uvm.o))

PAR_SEDOV_CPU_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.pcpu.o)
PAR_SEDOV_DEBUG_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.pdebug.o)
PAR_SEDOV_CUDA_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.pcuda.o))
PAR_SEDOV_CUDA_UVM_OBJ_FILES = $(if $(USE_CUDA),$(SEDOV_FILES:$(SRC)%.cpp=%.pcuda_uvm.o))

SEQ_UNIT_TESTS = unit_tests $(SEQ_SEDOV_TESTS)
SEQ_UNIT_TESTS += $(if $(USE_CUDA),cunit_tests)
PAR_UNIT_TESTS = punit_tests $(PAR_SEDOV_TESTS)

# Ceed tests
USE_CEED := $(MFEM_USE_CEED:NO=)
CEED_OBJ = $(CEED_SOURCE_FILES:$(SRC)%.cpp=%.o)
SEQ_UNIT_TESTS += $(if $(USE_CEED),ceed_tests)

ifeq ($(MFEM_USE_MPI),NO)
   UNIT_TESTS = $(SEQ_UNIT_TESTS)
else
   UNIT_TESTS = $(PAR_UNIT_TESTS) $(SEQ_UNIT_TESTS)
endif

all: $(UNIT_TESTS)

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: all clean

unit_tests: $(SEQ_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(SEQ_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

punit_tests: $(PAR_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(PAR_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

cunit_tests: $(CUDA_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(CUDA_MAIN_OBJ) $(OBJECT_FILES) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

ceed_tests: $(CEED_OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(CEED_OBJ) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

# Note: in this rule, we always use the full path to the source file as a
# workaround for an issue with coveralls.
$(OBJECT_FILES) $(SEQ_MAIN_OBJ) $(PAR_MAIN_OBJ) $(CUDA_MAIN_OBJ): %.o: $(SRC)%.cpp \
 $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) -c $(abspath $(<)) $(MFEM_FLAGS) $(INCLUDES) -o $(@)

$(CEED_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	$(CCC) -c $(abspath $(<)) $(MFEM_FLAGS) $(INCLUDES) -o $(@)

# Template rule for devices
# 1: postfix of 'sedov_tests_', 2: prefix of '_OBJECT_FILES',
# 3: configuration string for the MFEM device, separated with '$(comma)'
comma=,
define sedov_tests
sedov_tests_$(1): SEDOV_TESTS_FLAGS=-DMFEM_SEDOV_TESTS -DMFEM_SEDOV_DEVICE='"$(3)"'
sedov_tests_$(1): unit_test_main.o $$(SEQ_SEDOV_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(<) $$(SEQ_SEDOV_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(SEQ_SEDOV_$(2)_OBJ_FILES): %.$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) -c $$(abspath $$(<)) $$(MFEM_FLAGS) $$(INCLUDES) \
	   $$(SEDOV_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call sedov_tests,cpu,CPU,cpu))
$(eval $(call sedov_tests,debug,DEBUG,debug))
$(eval $(call sedov_tests,cuda,CUDA,cuda))
$(eval $(call sedov_tests,cuda_uvm,CUDA_UVM,cuda:uvm))

define psedov_tests
psedov_tests_$(1): SEDOV_TESTS_FLAGS =-DMFEM_SEDOV_MPI
psedov_tests_$(1): SEDOV_TESTS_FLAGS+=-DMFEM_SEDOV_TESTS -DMFEM_SEDOV_DEVICE='"$(3)"'
psedov_tests_$(1): punit_test_main.o $$(PAR_SEDOV_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(<) $$(PAR_SEDOV_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(PAR_SEDOV_$(2)_OBJ_FILES): %.p$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) -c $$(abspath $$(<)) $$(MFEM_FLAGS) $$(INCLUDES) \
	   $$(SEDOV_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call psedov_tests,cpu,CPU,cpu))
$(eval $(call psedov_tests,debug,DEBUG,debug))
$(eval $(call psedov_tests,cuda,CUDA,cuda))
$(eval $(call psedov_tests,cuda_uvm,CUDA_UVM,cuda:uvm))

$(DATA_DIR):
	ln -s $(SRC)$(DATA_DIR) .

# Testing
MFEM_TESTS = UNIT_TESTS
include $(MFEM_TEST_MK)

%-test-seq: %
	@$(call mfem-test,$<,, Unit tests,,SKIP-NO-VIS)

ceed_tests-test-seq: ceed_tests
	@$(call mfem-test,$<,, CEED Unit tests (cpu),--device ceed-cpu,SKIP-NO-VIS)
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-ref),--device ceed-cuda:/gpu/cuda/ref,SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-shared),--device ceed-cuda:/gpu/cuda/shared,SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-gen),--device ceed-cuda:/gpu/cuda/gen,SKIP-NO-VIS)
endif

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
