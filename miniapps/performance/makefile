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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/performance/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Distinguish x86 from PowerPC systems
MFEM_MACHINE ?= $(shell uname -m)

# Choose the switch MFEM_PERF_SW: gcc_x86_64, gcc_ppc64, clang, etc.
# The value of MFEM_PERF_SW is used to select MFEM_PERF_CXXFLAGS below.
define cxx_detect
cxx_v="$$($(MFEM_HOST_CXX) --version -c 2>&1)";
if [ 0 -ne $$? ]; then
  cxx_id="unknown";
elif [ -z "$${cxx_v##g++*}" ]; then
  cxx_id="gcc";
elif [ -z "$${cxx_v##*clang version*}" -o -z "$${cxx_v##*LLVM version*}" ]; then
  cxx_id="clang";
elif [ -z "$${cxx_v##*icpc*}" ]; then
    cxx_id="icc";
elif [ -z "$${cxx_v##*IBM XL*}" ]; then
  cxx_id="xlc";
elif [ -z "$${cxx_v##*pgc++*}" ]; then
  cxx_id="pgi";
else
  cxx_id="unknown";
fi;
printf "%s" "$$cxx_id"
endef
define DETECT_PERF_CXXFLAGS
ifneq (,$$(MFEM_HOST_CXX))
   MFEM_PERF_SW := $$(shell $$(cxx_detect))
   $$(info Detected host compiler: $$(MFEM_PERF_SW))
   ifeq (unknown,$$(MFEM_PERF_SW))
      $$(info -------------------------------------------)
      $$(info Output from '$$(MFEM_HOST_CXX) --version -c')
      $$(info -------------------------------------------)
      $$(shell $$(MFEM_HOST_CXX) --version -c 1>&2)
      $$(info -------------------------------------------)
   endif
endif
ifeq (gcc,$$(MFEM_PERF_SW))
   ifeq ($$(MFEM_MACHINE),x86_64)
      MFEM_PERF_SW = gcc_x86_64
   else ifneq (,$$(findstring ppc,$$(MFEM_MACHINE)))
      MFEM_PERF_SW = gcc_ppc
   endif
endif
# Choose MFEM_PERF_CXXFLAGS based on MFEM_PERF_SW:
MFEM_PERF_CXXFLAGS = $$(MFEM_PERF_CXXFLAGS_$$(MFEM_PERF_SW))
# Add MFEM_PERF_CXXFLAGS to MFEM_CXXFLAGS:
ifeq (YES,$$(MFEM_USE_CUDA))
  ifneq (,$$(MFEM_PERF_CXXFLAGS))
    ifneq ($$(MFEM_HOST_CXX),$$(MFEM_CXX))
      # using nvcc
      MFEM_CXXFLAGS += -Xcompiler="$$(MFEM_PERF_CXXFLAGS)"
    else
      # using clang+cuda
      MFEM_CXXFLAGS += $$(MFEM_PERF_CXXFLAGS)
    endif
  endif
else
  MFEM_CXXFLAGS += $$(MFEM_PERF_CXXFLAGS)
endif
DETECT_PERF_CXXFLAGS_DONE = YES
endef

# Compiler specific optimizations.
# For best performance, GCC 5 (or newer) is recommended.

ifneq (YES,$(MFEM_USE_CUDA))
  PEDANTIC_FLAG = -pedantic
endif

# - GCC extra options:
MFEM_PERF_CXXFLAGS_gcc_common += $(PEDANTIC_FLAG) -Wall
ifeq ($(MFEM_USE_SIMD),NO)
  MFEM_PERF_CXXFLAGS_gcc_common += --param max-completely-peel-times=3
endif
#MFEM_PERF_CXXFLAGS_gcc_common += -fdump-tree-optimized-blocks
MFEM_PERF_CXXFLAGS_gcc_x86_64 = -march=native $(MFEM_PERF_CXXFLAGS_gcc_common)
MFEM_PERF_CXXFLAGS_gcc_ppc = -mcpu=native -mtune=native\
 $(MFEM_PERF_CXXFLAGS_gcc_common)

# - XLC extra options:
MFEM_PERF_CXXFLAGS_xlc = -mcpu=native

# - Clang extra options:
ifeq ($(MFEM_MACHINE),riscv64)
  MFEM_PERF_CXXFLAGS_clang += -march=rv64gc
else ifneq (,$(findstring ppc,$(MFEM_MACHINE)))
  MFEM_PERF_CXXFLAGS_clang += -mcpu=native -mtune=native
else ifeq ($(MFEM_MACHINE),arm64)
  MFEM_PERF_CXXFLAGS_clang += -mcpu=native -mtune=native
else
  MFEM_PERF_CXXFLAGS_clang += -march=native
endif
MFEM_PERF_CXXFLAGS_clang += $(PEDANTIC_FLAG) -Wall
MFEM_PERF_CXXFLAGS_clang += -fcolor-diagnostics
MFEM_PERF_CXXFLAGS_clang += -fvectorize
MFEM_PERF_CXXFLAGS_clang += -fslp-vectorize
MFEM_PERF_CXXFLAGS_clang += -ffp-contract=fast

# - Intel C++ compiler extra options:
MFEM_PERF_CXXFLAGS_icc += -xHost


SEQ_MINIAPPS = ex1
PAR_MINIAPPS = ex1p
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(if $(DETECT_PERF_CXXFLAGS_DONE),,$(eval $(DETECT_PERF_CXXFLAGS)))
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), Performance miniapp,-rs 2)
ex1-test-seq: ex1
	@$(call mfem-test,$<,, Performance miniapp,-r 2)

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ ex1 ex1p
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh mesh.* sol.*
