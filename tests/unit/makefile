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

MAIN_OBJ = unit_test_main.o

USE_CUDA := $(MFEM_USE_CUDA:NO=)
USE_HIP := $(MFEM_USE_HIP:NO=)
USE_GPU := $(or $(USE_CUDA),$(USE_HIP))

# Each miniapp has serial/MPI executables with run-time device selection.
SEQ_SEDOV_TESTS = sedov_tests
PAR_SEDOV_TESTS = psedov_tests
SEQ_TMOP_TESTS = tmop_pa_tests
PAR_TMOP_TESTS = ptmop_pa_tests
SEQ_UNIT_TESTS = unit_tests $(SEQ_SEDOV_TESTS) $(SEQ_TMOP_TESTS)
PAR_UNIT_TESTS = unit_tests $(PAR_SEDOV_TESTS) $(PAR_TMOP_TESTS)

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

unit_tests: $(MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(MAIN_OBJ) $(LIBTESTS_O) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

ceed_tests: $(CEED_OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(CEED_OBJ) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

$(DEBUG_DEVICE_TEST): $(DEBUG_DEVICE_OBJ) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(CCC) $(DEBUG_DEVICE_OBJ) $(MFEM_LINK_FLAGS) $(MFEM_LIBS) -o $(@)

$(LIBTESTS_O): $(OBJECT_FILES)
	$(LD) -r $(OBJECT_FILES) -o $(@)

# Note: in this rule, we always use the full path to the source file as a
# workaround for an issue with coveralls.
$(OBJECT_FILES) $(MAIN_OBJ) \
 $(DEBUG_DEVICE_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) \
 $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) $(MFEM_FLAGS) $(INCLUDES) -c $(abspath $(<)) -o $(@)

$(MAIN_OBJ) $(DEBUG_DEVICE_OBJ): $(SRC)run_unit_tests.hpp

$(CEED_OBJ): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) $(MFEM_FLAGS) $(INCLUDES) -c $(abspath $(<)) -o $(@)

# Compile each miniapp once for each serial/MPI variant.
define miniapp_tests
$(1)_tests: miniapps/test_$(1).o $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$< $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$@
p$(1)_tests: miniapps/test_$(1).p.o $$(MFEM_LIB_FILE) $$(CONFIG_MK) $$(DATA_DIR)
	$$(CCC) $$< $$(MFEM_LINK_FLAGS) $$(MFEM_LIBS) -o $$@
miniapps/test_$(1).o: $$(SRC)miniapps/test_$(1).cpp $$(HEADER_FILES) $$(SRC)run_unit_tests.hpp $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -c $$(abspath $$<) -o $$@
miniapps/test_$(1).p.o: $$(SRC)miniapps/test_$(1).cpp $$(HEADER_FILES) $$(SRC)run_unit_tests.hpp $$(CONFIG_MK)
	@mkdir -p $$(@D)
	$$(CCC) $$(MFEM_FLAGS) $$(INCLUDES) -DMFEM_$(2)_MPI -c $$(abspath $$<) -o $$@
endef
$(eval $(call miniapp_tests,sedov,SEDOV))
$(eval $(call miniapp_tests,tmop_pa,TMOP_PA))

# For out-of-source builds, copy the meshes in ../../miniapps/multidomain from
# the source location; these are used by parallel 'unit_tests' runs.
# Also, link the directory 'meshes' in ../../miniapps/nurbs from the source
# location; this is needed by 'unit_tests'.
ifneq ($(SRC),)
.PHONY: copy-miniapps-multidomain-data copy-miniapps-nurbs-meshes
copy-miniapps-multidomain-data:
	$(MAKE) -C ../../miniapps/multidomain copy-data
unit_tests: | copy-miniapps-multidomain-data
copy-miniapps-nurbs-meshes:
	$(MAKE) -C ../../miniapps/nurbs copy-data
unit_tests: | copy-miniapps-nurbs-meshes
endif

# For out-of-source builds, copy the meshes in ../../miniapps/meshing from the
# source location; these are used by the TMOP tests.
.PHONY: copy-miniapps-meshing-data
copy-miniapps-meshing-data:
	$(MAKE) -C ../../miniapps/meshing copy-data

$(SEQ_TMOP_TESTS) $(PAR_TMOP_TESTS): | $(if $(SRC),copy-miniapps-meshing-data)

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

$(SEQ_SEDOV_TESTS:%=%-test-seq) $(SEQ_TMOP_TESTS:%=%-test-seq) $(DEBUG_DEVICE_TEST:%=%-test-seq): MFEM_DATA_FLAG =
$(PAR_SEDOV_TESTS:%=%-test-par) $(PAR_TMOP_TESTS:%=%-test-par): MFEM_DATA_FLAG =

unit_tests-test-seq: unit_tests
	@$(call mfem-test,$<,, Unit tests (cpu),--device cpu $(MFEM_DATA_FLAG),SKIP-NO-VIS)

# Run the same executable with each enabled device backend.
UNIT_TEST_DEVICES = $(if $(USE_GPU),gpu)
UNIT_TEST_DEVICES += $(if $(and $(USE_GPU),$(filter YES,$(MFEM_USE_RAJA))),raja-gpu)
UNIT_TEST_DEVICES += $(if $(filter YES,$(MFEM_USE_OPENMP)),omp)
ifeq ($(MFEM_USE_OPENMP)$(MFEM_USE_RAJA),YESYES)
# Inspect RAJA's own configuration rather than assuming it matches MFEM's.
RAJA_USE_OPENMP := $(filter YES,$(shell printf '\043include <RAJA/config.hpp>\n\043ifdef RAJA_ENABLE_OPENMP\nYES\n\043endif\n' | $(CCC) $(MFEM_FLAGS) -E -x c++ - 2>/dev/null))
UNIT_TEST_DEVICES += $(if $(RAJA_USE_OPENMP),raja-omp)
endif
test-par-NO: $(UNIT_TEST_DEVICES:%=unit_tests-test-seq-%)
test-par-YES: $(UNIT_TEST_DEVICES:%=unit_tests-test-seq-%) \
              $(UNIT_TEST_DEVICES:%=unit_tests-test-par-%)

unit_tests-test-seq-%: unit_tests
	@$(call mfem-test,$<,, Unit tests ($*),--device $* $(MFEM_DATA_FLAG),SKIP-NO-VIS,unit_tests-$*)

unit_tests-test-par-%: unit_tests
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel unit tests ($*),--parallel --device $* $(MFEM_DATA_FLAG),SKIP-NO-VIS,unit_tests-$*-np1)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel unit tests ($*),--parallel --device $* $(MFEM_DATA_FLAG),SKIP-NO-VIS,unit_tests-$*-np$(MFEM_MPI_NP))

SEDOV_TEST_DEVICES = $(UNIT_TEST_DEVICES) debug $(if $(USE_GPU),gpu_uvm)
TMOP_TEST_DEVICES = $(UNIT_TEST_DEVICES) debug
PAR_TMOP_TEST_DEVICES = $(UNIT_TEST_DEVICES)
test-par-NO: $(SEDOV_TEST_DEVICES:%=sedov_tests-test-seq-%) \
              $(TMOP_TEST_DEVICES:%=tmop_pa_tests-test-seq-%)
test-par-YES: $(SEDOV_TEST_DEVICES:%=sedov_tests-test-seq-%) \
               $(TMOP_TEST_DEVICES:%=tmop_pa_tests-test-seq-%) \
               $(SEDOV_TEST_DEVICES:%=psedov_tests-test-par-%) \
               $(PAR_TMOP_TEST_DEVICES:%=ptmop_pa_tests-test-par-%)

sedov_tests-test-seq-%: sedov_tests
	@$(call mfem-test,$<,, Sedov tests ($*),--device $(subst _,:,$*),SKIP-NO-VIS,sedov_tests-$*)
tmop_pa_tests-test-seq-%: tmop_pa_tests
	@$(call mfem-test,$<,, TMOP tests ($*),--device $*,SKIP-NO-VIS,tmop_pa_tests-$*)
psedov_tests-test-par-%: psedov_tests
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel Sedov tests ($*),--device $(subst _,:,$*),SKIP-NO-VIS,psedov_tests-$*-np1)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel Sedov tests ($*),--device $(subst _,:,$*),SKIP-NO-VIS,psedov_tests-$*-np$(MFEM_MPI_NP))
ptmop_pa_tests-test-par-%: ptmop_pa_tests
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel TMOP tests ($*),--device $*,SKIP-NO-VIS,ptmop_pa_tests-$*-np1)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel TMOP tests ($*),--device $*,SKIP-NO-VIS,ptmop_pa_tests-$*-np$(MFEM_MPI_NP))

ceed_tests-test-seq: ceed_tests
	@$(call mfem-test,$<,, CEED Unit tests (cpu),--device ceed-cpu $(MFEM_DATA_FLAG),SKIP-NO-VIS)
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-ref),--device ceed-cuda:/gpu/cuda/ref $(MFEM_DATA_FLAG),SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-shared),--device ceed-cuda:/gpu/cuda/shared $(MFEM_DATA_FLAG),SKIP-NO-VIS)
	@$(call mfem-test,$<,, CEED Unit tests (cuda-gen),--device ceed-cuda:/gpu/cuda/gen $(MFEM_DATA_FLAG),SKIP-NO-VIS)
endif

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP)
unit_tests-test-par: MFEM_PARALLEL_FLAG = --parallel
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel unit tests,$(MFEM_PARALLEL_FLAG) $(MFEM_DATA_FLAG),SKIP-NO-VIS,$<-np1)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel unit tests,$(MFEM_PARALLEL_FLAG) $(MFEM_DATA_FLAG),SKIP-NO-VIS,$<-np$(MFEM_MPI_NP))

unit_tests-test-par: unit_tests
	@$(call mfem-test,$<, $(RUN_MPI) 1, Parallel unit tests (cpu),--parallel --device cpu $(MFEM_DATA_FLAG),SKIP-NO-VIS,unit_tests-cpu-np1)
	@$(call mfem-test,$<, $(RUN_MPI) $(MFEM_MPI_NP), Parallel unit tests (cpu),--parallel --device cpu $(MFEM_DATA_FLAG),SKIP-NO-VIS,unit_tests-cpu-np$(MFEM_MPI_NP))

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean-exec:
	@rm -rf output_meshes parallel_in_serial.mesh parallel_in_serial.gf

clean: clean-exec
	rm -f $(SEQ_UNIT_TESTS) $(PAR_UNIT_TESTS) \
	   *.o */*.o */*~ *~
	rm -rf *.dSYM
