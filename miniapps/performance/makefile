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

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/performance/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Distinguish x86 from PowerPC systems
MFEM_MACHINE ?= $(shell uname -m)

# Choose the switch MFEM_PERF_SW: gcc_x86_64, gcc_ppc64, clang, or icc.
# The value of MFEM_PERF_SW is used to select MFEM_PERF_CXXFLAGS below.
ifneq (,$(MFEM_PERF_SW))
   # Use the value of MFEM_PERF_SW if already defined
else ifneq (,$(filter %clang++ %mpiclang++,$(MFEM_CXX)))
   MFEM_PERF_SW = clang
else ifneq (,$(filter %g++ %mpicxx %mpic++,$(MFEM_CXX)))
   ifeq ($(MFEM_MACHINE),x86_64)
      MFEM_PERF_SW = gcc_x86_64
   else ifneq (,$(findstring ppc64,$(MFEM_MACHINE)))
      MFEM_PERF_SW = gcc_ppc64
   endif
else ifneq (,$(filter %icpc %mpiicpc,$(MFEM_CXX)))
   MFEM_PERF_SW = icc
endif

# Compiler specific optimizations.
# For best performance, GCC 5 (or newer) is recommended.

# - GCC extra options:
# MFEM_PERF_CXXFLAGS_gcc_common += -std=c++03
MFEM_PERF_CXXFLAGS_gcc_common += -std=c++11
MFEM_PERF_CXXFLAGS_gcc_common += -pedantic -Wall
ifeq ($(MFEM_USE_X86INTRIN),NO)
MFEM_PERF_CXXFLAGS_gcc_common += --param max-completely-peel-times=3
endif
#MFEM_PERF_CXXFLAGS_gcc_common += -fdump-tree-optimized-blocks
MFEM_PERF_CXXFLAGS_gcc_x86_64 = -march=native $(MFEM_PERF_CXXFLAGS_gcc_common)
MFEM_PERF_CXXFLAGS_gcc_ppc64 = -mcpu=native -mtune=native\
 $(MFEM_PERF_CXXFLAGS_gcc_common)

# - Clang extra options:
MFEM_PERF_CXXFLAGS_clang += -march=native
# MFEM_PERF_CXXFLAGS_clang += -std=c++03
MFEM_PERF_CXXFLAGS_clang += -std=c++11
MFEM_PERF_CXXFLAGS_clang += -pedantic -Wall
MFEM_PERF_CXXFLAGS_clang += -fcolor-diagnostics
MFEM_PERF_CXXFLAGS_clang += -fvectorize
MFEM_PERF_CXXFLAGS_clang += -fslp-vectorize
MFEM_PERF_CXXFLAGS_clang += -fslp-vectorize-aggressive
MFEM_PERF_CXXFLAGS_clang += -ffp-contract=fast

# - Intel C++ compiler extra options:
MFEM_PERF_CXXFLAGS_icc += -xHost
# MFEM_PERF_CXXFLAGS_icc += -std=c++03
MFEM_PERF_CXXFLAGS_icc += -std=c++11

# Choose MFEM_PERF_CXXFLAGS based on MFEM_PERF_SW:
MFEM_PERF_CXXFLAGS = $(MFEM_PERF_CXXFLAGS_$(MFEM_PERF_SW))
# Add MFEM_PERF_CXXFLAGS to MFEM_CXXFLAGS:
MFEM_CXXFLAGS += $(MFEM_PERF_CXXFLAGS)

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

# ******************************************************************************
# PERF TESTS GEN/GO/GET
# ******************************************************************************
PWD = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
PERF_PATH = perf
REF_PATH = /home/camier1/home/mfem/master/miniapps/performance/perf
MAKEFILE_DUMP  = MFEM_DIR=$(PWD)/../..\\nMFEM_BUILD_DIR=$(PWD)/../..\\n
MAKEFILE_DUMP += \\nMFEM_PERF_CXXFLAGS_gcc_common+=-DMESH_P=$$order 
MAKEFILE_DUMP += -DSOL_P=$$order -DPROBLEM=$$problem -DMESH_FILE=\\\"$(MESH_FILE)\\\"\\n
MAKEFILE_DUMP += \\ninclude $(shell pwd)/makefile
ORDERs = 2 3 4 5 6 7 8 9 10 11 12 13 14 #15 16
PROBLEMs = 0 1
SREFINEs = 1 2 3 #4
PREFINEs = 0
MESH_FILE = $(PWD)/../../data/inline-hex-2x1x1.mesh
ECHO=/bin/echo
make_bp:;@make bp
gen:
	mkdir -v -p $(PERF_PATH)
	@tput reset
	@echo Now launching GEN
	@for problem in $(PROBLEMs);do\
		for order in $(ORDERs);do\
			$(ECHO) -e \\tpb$$problem\_o$$order;\
			path=$(PERF_PATH)/pb$$problem/o$$order;\
			mkdir -v -p $$path;\
			ln -fs $(shell pwd)/bp.cpp $$path;\
			$(ECHO) -e $(MAKEFILE_DUMP) > $$path/makefile;\
			(cd $$path && make make_bp && mv bp bp_pb$$problem\_o$$order);\
		done;\
	done

go:
	@tput reset
	@echo Now launching tests
	@for problem in $(PROBLEMs);do\
		for order in $(ORDERs);do\
			path=$(PERF_PATH)/pb$$problem/o$$order;\
			exec=bp"_pb"$$problem"_o"$$order;\
			for rs in $(SREFINEs);do\
				for rp in $(PREFINEs);do\
					rsp=$$(($$rs+$$rp));\
					$(ECHO) -e \\tpb$$problem\_o$$order\_rs$$rs\_rp$$rp;\
					file=bp"_pb"$$problem"_o"$$order"_rs"$$rs"_rp"$$rp;\
					perf stat -e instructions,cycles,task-clock,cpu-clock \
-o $$path/$$file.perf $$path/$$exec -rs $$rs -rp $$rp > $$path/$$file.out;\
				done;\
			done;\
		done;\
	done

get:
	@tput reset && sync && sync
	@echo Now collecting results
	@for problem in $(PROBLEMs);do\
		path=$(PERF_PATH)/pb$$problem;\
		OUTPUT_FILE=mfem-pb$$problem\.org;\
		$(ECHO) -e \\tGenerating: $$OUTPUT_FILE;\
		$(ECHO) -n > $$OUTPUT_FILE;\
		$(ECHO) \#\|order\|refine\|one\|perf\| >> $$OUTPUT_FILE &&\
		for order in $(ORDERs);do\
			$(ECHO) -e \\t\\tORDER=$$order;\
			opath=$$path/o$$order;\
			for rs in $(SREFINEs);do\
				for rp in $(PREFINEs);do\
					rsp=$$(($$rs+$$rp));\
					REF_VAL=`cat $(REF_PATH)/pb$$problem/o$$order/bp_pb$$problem\_o$$order\_rs$$rs\_rp$$rp\.out|grep DOFs|grep CG|cut -d' ' -f4|tr -d [:blank:]`;\
					$(ECHO) -e \\t\\t\\trefine=$$rsp;\
					($(ECHO) -n \|$$order\|$$rsp\|$$REF_VAL\| >> $$OUTPUT_FILE &&\
					cat $(PWD)/$$opath/bp_pb$$problem\_o$$order\_rs$$rs\_rp$$rp\.out|grep DOFs|grep CG|cut -d' ' -f4|tr -d [:blank:]|tr -d \\n >> $$OUTPUT_FILE &&\
					$(ECHO) \| >> $$OUTPUT_FILE);\
				done;\
			done;\
			$(ECHO) >> $$OUTPUT_FILE;\
		done;\
	done

# ******************************************************************************
GEOX ?= 3072
GEOY ?= 2048
FNTS ?= 4
VIEWX ?= 40 #120
VIEWY ?= 20
FILE ?= mfem-pb1.org
FILE_P0 ?= mfem-pb0.org
FILE_P1 ?= mfem-pb1.org
XMAX ?= 4
YSTP ?= 1
YMAX ?= 3
ZMIN ?= 1
ZMAX ?= 3.5
CNTD ?= 2
TITLE ?= Total Rate
NUM ?= 20
DNM ?= 10
CPU = $(shell echo $(shell getconf _NPROCESSORS_ONLN))

gp gnuplot:
	$(MAKE) -j $(CPU) bp0 bp1

bp0: $(FILE_P0)
	FILE=$(FILE_P0) \
	TITLE="BP 3D P0 Speedup: x86\/master" \
	NUM=5 DNM=4 ZMAX= $(MAKE) mfem-$@.png

bp1: $(FILE_P1)
	FILE=$(FILE_P1) \
	TITLE="BP 3D P1 Speedup: x86\/master" \
	NUM=5 DNM=4 ZMAX= $(MAKE) mfem-$@.png

#.PRECIOUS: %.gpi
%.gpi: makefile bp.tgp
	sed 's/TITLE/$(TITLE)/g;s/XMAX/$(XMAX)/g;s/CNTD/$(CNTD)/g;s/YMAX/$(YMAX)/g;s/YSTP/$(YSTP)/g;s/ZMIN/$(ZMIN)/g;s/ZMAX/$(ZMAX)/g;s/NUM/$(NUM)/g;s/DNM/$(DNM)/g;s/GEOX/$(GEOX)/g;s/GEOY/$(GEOY)/g;s/FNTS/$(FNTS)/g;s/VIEWX/$(VIEWX)/g;s/VIEWY/$(VIEWY)/g;s/FILE/$(FILE)/g' bp.tgp > $*.gpi

%.png:%.gpi $(FILE)
	gnuplot $*.gpi > $*.png

%.pdf:%.png
	convert $(CONVERT_FLAGS) $*.png $*.pdf
