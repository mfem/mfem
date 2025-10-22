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

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/tests/unit/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

CC = $(MFEM_CXX)
CCOPTS = -g
CCC = $(CC) $(CCOPTS)

# -I$(MFEM_DIR) is needed by some tests, e.g. to #include "general/text.hpp"
INCLUDES = -I$(or $(SRC:%/=%),.) -I$(MFEM_DIR)

SOURCE_FILES := $(sort $(wildcard $(SRC)*/*.cpp))
CEED_SOURCE_FILES = $(sort $(wildcard $(SRC)ceed/*.cpp))
MINI_SOURCE_FILES = $(sort $(wildcard $(SRC)miniapps/*.cpp))
SOURCE_FILES := $(filter-out \
   $(CEED_SOURCE_FILES) $(MINI_SOURCE_FILES), $(SOURCE_FILES))
HEADER_FILES = $(SRC)catch.hpp $(SRC)unit_tests.hpp
OBJECT_FILES = $(SOURCE_FILES:$(SRC)%.cpp=%.o)
LIBTESTS_O = libtests.o
DATA_DIR = data

SEQ_MAIN_OBJ = unit_test_main.o
PAR_MAIN_OBJ = punit_test_main.o
GPU_MAIN_OBJ = gpu_unit_test_main.o
PGPU_MAIN_OBJ = pgpu_unit_test_main.o

# Sedov numerical seq/par files and tests
SEDOV_FILES = $(SRC)miniapps/test_sedov.cpp

USE_CUDA := $(MFEM_USE_CUDA:NO=)
USE_HIP := $(MFEM_USE_HIP:NO=)
USE_GPU := $(or $(USE_CUDA),$(USE_HIP))
SEQ_SEDOV_TESTS = sedov_tests_cpu sedov_tests_debug
SEQ_SEDOV_TESTS += $(if $(USE_GPU),sedov_tests_gpu)
SEQ_SEDOV_TESTS += $(if $(USE_GPU),sedov_tests_gpu_uvm)
PAR_SEDOV_TESTS = $(SEQ_SEDOV_TESTS:%=p%)

SEQ_SEDOV_CPU_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.cpu.o)
SEQ_SEDOV_DEBUG_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.debug.o)
SEQ_SEDOV_GPU_OBJ_FILES = $(if $(USE_GPU),$(SEDOV_FILES:$(SRC)%.cpp=%.gpu.o))
SEQ_SEDOV_GPU_UVM_OBJ_FILES = $(if $(USE_GPU),$(SEDOV_FILES:$(SRC)%.cpp=%.gpu_uvm.o))

PAR_SEDOV_CPU_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.pcpu.o)
PAR_SEDOV_DEBUG_OBJ_FILES = $(SEDOV_FILES:$(SRC)%.cpp=%.pdebug.o)
PAR_SEDOV_GPU_OBJ_FILES = $(if $(USE_GPU),$(SEDOV_FILES:$(SRC)%.cpp=%.pgpu.o))
PAR_SEDOV_GPU_UVM_OBJ_FILES = $(if $(USE_GPU),$(SEDOV_FILES:$(SRC)%.cpp=%.pgpu_uvm.o))

# TMOP numerical seq/par files and tests
TMOP_FILES = $(SRC)miniapps/test_tmop_pa.cpp

SEQ_TMOP_TESTS = tmop_pa_tests_cpu tmop_pa_tests_debug
SEQ_TMOP_TESTS += $(if $(USE_GPU),tmop_pa_tests_gpu)
# SEQ_TMOP_TESTS += $(if $(USE_GPU),tmop_tests_gpu_uvm)
PAR_TMOP_TESTS = ptmop_pa_tests_cpu
PAR_TMOP_TESTS += $(if $(USE_GPU),ptmop_pa_tests_gpu)

SEQ_TMOP_CPU_OBJ_FILES = $(TMOP_FILES:$(SRC)%.cpp=%.cpu.o)
SEQ_TMOP_DEBUG_OBJ_FILES = $(TMOP_FILES:$(SRC)%.cpp=%.debug.o)
SEQ_TMOP_GPU_OBJ_FILES = $(if $(USE_GPU),$(TMOP_FILES:$(SRC)%.cpp=%.gpu.o))
SEQ_TMOP_GPU_UVM_OBJ_FILES = $(if $(USE_GPU),$(TMOP_FILES:$(SRC)%.cpp=%.gpu_uvm.o))

PAR_TMOP_CPU_OBJ_FILES = $(TMOP_FILES:$(SRC)%.cpp=%.pcpu.o)
PAR_TMOP_DEBUG_OBJ_FILES = $(TMOP_FILES:$(SRC)%.cpp=%.pdebug.o)
PAR_TMOP_GPU_OBJ_FILES = $(if $(USE_GPU),$(TMOP_FILES:$(SRC)%.cpp=%.pgpu.o))
PAR_TMOP_GPU_UVM_OBJ_FILES = $(if $(USE_GPU),$(TMOP_FILES:$(SRC)%.cpp=%.pgpu_uvm.o))

# seq/par files and tests
SEQ_UNIT_TESTS = unit_tests $(if $(USE_GPU),gpu_unit_tests)
SEQ_UNIT_TESTS += $(SEQ_SEDOV_TESTS) $(SEQ_TMOP_TESTS)
PAR_UNIT_TESTS = punit_tests $(if $(USE_GPU),pgpu_unit_tests)
PAR_UNIT_TESTS += $(PAR_SEDOV_TESTS) $(PAR_TMOP_TESTS)

# Ceed tests
USE_CEED := $(MFEM_USE_CEED:NO=)
CEED_OBJ = $(CEED_SOURCE_FILES:$(SRC)%.cpp=%.o)
SEQ_UNIT_TESTS += $(if $(USE_CEED),ceed_tests)

# Debug device tests
DEBUG_DEVICE_FILE = $(SRC)miniapps/test_debug_device.cpp
DEBUG_DEVICE_OBJ = $(DEBUG_DEVICE_FILE:$(SRC)%.cpp=%.o)
DEBUG_DEVICE_TEST = debug_device_tests
SEQ_UNIT_TESTS += $(DEBUG_DEVICE_TEST)

ifeq ($(MFEM_USE_MPI),NO)
   UNIT_TESTS = $(SEQ_UNIT_TESTS)
else
   UNIT_TESTS = $(PAR_UNIT_TESTS) $(SEQ_UNIT_TESTS)
endif

all: $(UNIT_TESTS)

.SUFFIXES:
.SUFFIXES: .cpp .o
.PHONY: all clean clean-exec

unit_tests: $(SEQ_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(SEQ_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

punit_tests: $(PAR_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(PAR_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

gpu_unit_tests: $(GPU_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(GPU_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

pgpu_unit_tests: $(PGPU_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(PGPU_MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

ceed_tests: $(CEED_OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(CEED_OBJ) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

$(DEBUG_DEVICE_TEST): $(DEBUG_DEVICE_OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(CCC) $(DEBUG_DEVICE_OBJ) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

$(LIBTESTS_O): $(OBJECT_FILES)
	$(LD) -r $(OBJECT_FILES) -o $(@)

# Note: in this rule, we always use the full path to the source file as a
# workaround for an issue with coveralls.
$(OBJECT_FILES) $(SEQ_MAIN_OBJ) $(PAR_MAIN_OBJ) $(GPU_MAIN_OBJ) \
 $(PGPU_MAIN_OBJ) $(DEBUG_DEVICE_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) \
 $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) $(MFEM_FLAGS) $(INCLUDES) -c $(abspath $(<)) -o $(@)

$(CEED_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) $(MFEM_FLAGS) $(INCLUDES) -c $(abspath $(<)) -o $(@)

# Template rule for devices
# SEDOV Template rule for devices
# 1: postfix of 'sedov_tests_', 2: prefix of '_OBJECT_FILES',
# 3: configuration string for the MFEM device, separated with '$(comma)'
comma=,
define sedov_tests
sedov_tests_$(1): SEDOV_TESTS_FLAGS=-DMFEM_SEDOV_DEVICE='"$(3)"'
sedov_tests_$(1): $$(SEQ_SEDOV_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(SEQ_SEDOV_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(SEQ_SEDOV_$(2)_OBJ_FILES): %.$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -c $$(abspath $$(<)) \
	   $$(SEDOV_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call sedov_tests,cpu,CPU,cpu))
$(eval $(call sedov_tests,debug,DEBUG,debug))
$(eval $(call sedov_tests,gpu,GPU,gpu))
$(eval $(call sedov_tests,gpu_uvm,GPU_UVM,gpu:uvm))

define psedov_tests
psedov_tests_$(1): SEDOV_TESTS_FLAGS =-DMFEM_SEDOV_MPI
psedov_tests_$(1): SEDOV_TESTS_FLAGS+=-DMFEM_SEDOV_DEVICE='"$(3)"'
psedov_tests_$(1): $$(PAR_SEDOV_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(PAR_SEDOV_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(PAR_SEDOV_$(2)_OBJ_FILES): %.p$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -c $$(abspath $$(<)) \
	   $$(SEDOV_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call psedov_tests,cpu,CPU,cpu))
$(eval $(call psedov_tests,debug,DEBUG,debug))
$(eval $(call psedov_tests,gpu,GPU,gpu))
$(eval $(call psedov_tests,gpu_uvm,GPU_UVM,gpu:uvm))

# For out-of-source builds, copy the meshes in ../../miniapps/multidomain from
# the source location; these are used by 'punit_tests'.
# Also, link the directory 'meshes' in ../../miniapps/nurbs from the source
# location; this is needed by 'unit_tests'.
ifneq ($(SRC),)
.PHONY: copy-miniapps-multidomain-data copy-miniapps-nurbs-meshes
copy-miniapps-multidomain-data:
	$(MAKE) -C ../../miniapps/multidomain copy-data
punit_tests: | copy-miniapps-multidomain-data
copy-miniapps-nurbs-meshes:
	$(MAKE) -C ../../miniapps/nurbs copy-data
unit_tests: | copy-miniapps-nurbs-meshes
endif

# For out-of-source builds, copy the meshes in ../../miniapps/meshing from the
# source location; these are used by the TMOP tests.
.PHONY: copy-miniapps-meshing-data
copy-miniapps-meshing-data:
	$(MAKE) -C ../../miniapps/meshing copy-data

# TMOP Template rule for devices
# 1: postfix of 'tmop_tests_', 2: prefix of '_OBJECT_FILES',
# 3: configuration string for the MFEM device, separated with '$(comma)'
define tmop_pa_tests
tmop_pa_tests_$(1): | $(if $(SRC),copy-miniapps-meshing-data)
tmop_pa_tests_$(1): TMOP_TESTS_FLAGS=-DMFEM_TMOP_PA_DEVICE='"$(3)"'
tmop_pa_tests_$(1): $$(SEQ_TMOP_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(SEQ_TMOP_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(SEQ_TMOP_$(2)_OBJ_FILES): %.$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -c $$(abspath $$(<)) \
	   $$(TMOP_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call tmop_pa_tests,cpu,CPU,cpu))
$(eval $(call tmop_pa_tests,debug,DEBUG,debug))
$(eval $(call tmop_pa_tests,gpu,GPU,gpu))
# $(eval $(call tmop_pa_tests,gpu_uvm,GPU_UVM,gpu:uvm))

define ptmop_pa_tests
ptmop_pa_tests_$(1): | $(if $(SRC),copy-miniapps-meshing-data)
ptmop_pa_tests_$(1): TMOP_TESTS_FLAGS =-DMFEM_TMOP_MPI
ptmop_pa_tests_$(1): TMOP_TESTS_FLAGS+=-DMFEM_TMOP_PA_DEVICE='"$(3)"'
ptmop_pa_tests_$(1): $$(PAR_TMOP_$(2)_OBJ_FILES) \
 $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$(PAR_TMOP_$(2)_OBJ_FILES) \
	   $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$(@)
$$(PAR_TMOP_$(2)_OBJ_FILES): %.p$(1).o: $$(SRC)%.cpp $$(HEADER_FILES) $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -c $$(abspath $$(<)) \
	   $$(TMOP_TESTS_FLAGS) -o $$(@)
endef
$(eval $(call ptmop_pa_tests,cpu,CPU,cpu))
#$(eval $(call ptmop_pa_tests,debug,DEBUG,debug))
$(eval $(call ptmop_pa_tests,gpu,GPU,gpu))
#$(eval $(call ptmop_pa_tests,gpu_uvm,GPU_UVM,gpu:uvm))

$(DATA_DIR):
	ln -s $(SRC)$(DATA_DIR) .

# Testing
MFEM_TESTS = UNIT_TESTS
include $(MFEM_TEST_MK)

ifeq (,$(wildcard $(MFEM_DIR)/../data))
   MFEM_DATA_FLAG =
else
   MFEM_DATA_FLAG = --data $(MFEM_DIR)/../data
endif

%-test-seq: %
	@$(call mfem-test,$<,, Unit tests,$(MFEM_DATA_FLAG),SKIP-NO-VIS)

ceed_tests-test-seq: ceed_tests
	@$(call mfem-test,$<,, CEED Unit tests (cpu),--device ceed-cpu $(MFEM_DATA_FLAG),SKIP-NO-VIS)
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-ref),--device ceed-cuda:/gpu/cuda/ref $(MFEM_DATA_FLAG),SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-shared),--device ceed-cuda:/gpu/cuda/shared $(MFEM_DATA_FLAG),SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-gen),--device ceed-cuda:/gpu/cuda/gen $(MFEM_DATA_FLAG),SKIP-NO-VIS)
endif

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP)
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel unit tests,$(MFEM_DATA_FLAG),SKIP-NO-VIS)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel unit tests,$(MFEM_DATA_FLAG),SKIP-NO-VIS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean-exec:
	@rm -rf output_meshes parallel_in_serial.mesh parallel_in_serial.gf

clean: clean-exec
	rm -f $(SEQ_UNIT_TESTS) $(PAR_UNIT_TESTS) *.o */*.o */*~ *~
	rm -rf *.dSYM
