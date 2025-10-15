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
MFEM_DIR ?= ..
MFEM_BUILD_DIR ?= ..
MFEM_INSTALL_DIR ?= ../mfem
SRC = $(if $(MFEM_DIR:..=),$(MFEM_DIR)/examples/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex0 ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex14 ex15 ex16 \
   ex17 ex18 ex19 ex20 ex21 ex22 ex23 ex24 ex25 ex26 ex27 ex28 ex29 ex30 \
   ex31 ex33 ex34 ex36 ex37 ex38 ex39 ex40
PAR_EXAMPLES = ex0p ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex8p ex9p ex10p ex11p \
   ex12p ex13p ex14p ex15p ex16p ex17p ex18p ex19p ex20p ex21p ex22p ex24p \
   ex25p ex26p ex27p ex28p ex29p ex30p ex31p ex32p ex33p ex34p ex35p ex36p \
   ex37p ex39p ex40p
SEQ_DEVICE_EXAMPLES = ex1 ex3 ex4 ex5 ex6 ex9 ex14 ex22 ex24 ex25 ex26 ex34
PAR_DEVICE_EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex9p ex13p ex14p \
   ex22p ex24p ex25p ex26p ex34p ex35p

ifeq ($(MFEM_USE_LAPACK),YES)
   SEQ_EXAMPLES += ex38
endif

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif
SUBDIRS =
ifeq ($(MFEM_USE_AMGX),YES)
   SUBDIRS += amgx
endif
ifeq ($(MFEM_USE_GINKGO),YES)
   SUBDIRS += ginkgo
endif
ifeq ($(MFEM_USE_HIOP),YES)
   SUBDIRS += hiop
endif
ifeq ($(MFEM_USE_PETSC),YES)
   SUBDIRS += petsc
endif
ifeq ($(MFEM_USE_PUMI),YES)
   SUBDIRS += pumi
endif
ifeq ($(MFEM_USE_SUNDIALS),YES)
   SUBDIRS += sundials
endif
ifeq ($(MFEM_USE_SUPERLU),YES)
   SUBDIRS += superlu
endif
ifeq ($(MFEM_USE_MOONOLITH),YES)
   SUBDIRS += moonolith
endif
ifeq ($(MFEM_USE_CALIPER),YES)
   SUBDIRS += caliper
endif

SUBDIRS_ALL = $(addsuffix /all,$(SUBDIRS))
SUBDIRS_TEST = $(addsuffix /test,$(SUBDIRS))
SUBDIRS_CLEAN = $(addsuffix /clean,$(SUBDIRS))
SUBDIRS_TPRINT = $(addsuffix /test-print,$(SUBDIRS))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES) $(SUBDIRS_ALL)

.PHONY: $(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN) $(SUBDIRS_TPRINT)
$(SUBDIRS_ALL) $(SUBDIRS_TEST) $(SUBDIRS_CLEAN):
	$(MAKE) -C $(@D) $(@F)
$(SUBDIRS_TPRINT):
	@$(MAKE) -C $(@D) $(@F)

# Additional dependencies
ex18: $(SRC)ex18.hpp
ex33: $(SRC)ex33.hpp
ex37: $(SRC)ex37.hpp

ifeq ($(MFEM_USE_MPI),YES)
ex18p: $(SRC)ex18.hpp
ex33p: $(SRC)ex33.hpp
ex37p: $(SRC)ex37.hpp
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)
test: $(SUBDIRS_TEST)
test-print: $(SUBDIRS_TPRINT)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example)
%-test-seq: %
	@$(call mfem-test,$<,, Serial example)
%-test-par-cuda: %
	@$(call mfem-test,$<, $(RUN_MPI), Parallel CUDA example,-d cuda)
%-test-seq-cuda: %
	@$(call mfem-test,$<,, Serial CUDA example,-d cuda)
%-test-par-hip: %
	@$(call mfem-test,$<, $(RUN_MPI), Parallel HIP example,-d hip)
%-test-seq-hip: %
	@$(call mfem-test,$<,, Serial HIP example,-d hip)

# Testing: Specific execution options
ex0-test-seq: ex0
	@$(call mfem-test,$<,, Serial example,,1)
ex0p-test-par: ex0p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,,1)
ex1-test-seq: ex1
	@$(call mfem-test,$<,, Serial example)
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example)
ex10-test-seq: ex10
	@$(call mfem-test,$<,, Serial example,-tf 5)
ex10p-test-par: ex10p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-tf 5)
ex14-test-seq-cuda: ex14
	@$(call mfem-test,$<,, Serial CUDA example,-r 2 -pa -d cuda)
ex14p-test-par-cuda: ex14p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel CUDA example,-rs 2 -rp 0 -pa -d cuda)
ex14-test-seq-hip: ex14
	@$(call mfem-test,$<,, Serial HIP example,-r 2 -pa -d hip)
ex14p-test-par-hip: ex14p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel HIP example,-rs 2 -rp 0 -pa -d hip)
ex15-test-seq: ex15
	@$(call mfem-test,$<,, Serial example,-e 1)
ex15p-test-par: ex15p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-e 1)
ex27-test-seq: ex27
	@$(call mfem-test,$<,, Serial example,-dg)
ex27p-test-par: ex27p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-dg)
ex37-test-seq: ex37
	@$(call mfem-test,$<,, Serial example,-mi 3)
ex37p-test-par: ex37p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-mi 3)
# Testing: optional tests
ifeq ($(MFEM_USE_STRUMPACK),YES)
ex11p-test-strumpack: ex11p
	@$(call mfem-test,$<, $(RUN_MPI), STRUMPACK example,--strumpack)
test-par-YES: ex11p-test-strumpack
endif
ifeq ($(MFEM_USE_SUPERLU),YES)
ex11p-test-superlu: ex11p
	@$(call mfem-test,$<, $(RUN_MPI), SuperLU_DIST example,--superlu)
test-par-YES: ex11p-test-superlu
endif
ifeq ($(MFEM_USE_MKL_CPARDISO),YES)
ex11p-test-cpardiso: ex11p
	@$(call mfem-test,$<, $(RUN_MPI), MKL_CPARDISO example,--cpardiso)
test-par-YES: ex11p-test-cpardiso
endif
ifeq ($(MFEM_USE_CEED),YES)
ex1p-test-ceed: ex1p
	@$(call mfem-test,$<, $(RUN_MPI),\
	   Parallel libCEED example,-d ceed-cpu -pa -a)
test-par-YES: ex1p-test-ceed
endif

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec $(SUBDIRS_CLEAN)

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh displaced.mesh mesh.* ex5.mesh ex6p-checkpoint.*
	@rm -rf Example5* Example9* Example15* Example16* Example23* ParaView
	@rm -f sphere_refined.* sol.* sol_u.* sol_p.* sol_r.* sol_i.* order.*
	@rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
	@rm -f deformed.* velocity.* elastic_energy.* mode_* mode_deriv_* flux.*
	@rm -f ex5-p-*.bp ex9-p-*.bp ex12-p-*.bp ex16-p-*.bp
	@rm -f ex16.mesh ex16-mesh.* ex16-init.* ex16-final.*
	@rm -f deformation.* pressure.*
	@rm -f ex20.dat ex20p_?????.dat gnuplot_ex20.inp gnuplot_ex20p.inp
	@rm -f ex21*.mesh ex21*.sol ex21p_*.*
	@rm -f ex23.mesh ex23-*.gf
	@rm -f ex25.mesh ex25-*.gf ex25p-*.*
	@rm -f euler-?-final.* euler-?-init.* euler-mesh-final.* euler-mesh.*
	@rm -rf ex28_* ex28p_*
	@rm -rf cond.* cond_mesh.* cond_j.* dsol.* port_mesh.* port_mode.*
