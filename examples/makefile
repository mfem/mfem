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
MFEM_DIR ?= ..
MFEM_BUILD_DIR ?= ..
SRC = $(if $(MFEM_DIR:..=),$(MFEM_DIR)/examples/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex14 ex15 ex16 ex17\
  ex18 ex19 ex20 ex21 ex22
PAR_EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex8p ex9p ex10p ex11p ex12p\
 ex13p ex14p ex15p ex16p ex17p ex18p ex19p ex20p ex21p ex22p

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif
SUBDIRS =
ifeq ($(MFEM_USE_SUNDIALS),YES)
   SUBDIRS += sundials
endif
ifeq ($(MFEM_USE_PETSC),YES)
   SUBDIRS += petsc
endif
ifeq ($(MFEM_USE_PUMI),YES)
   SUBDIRS += pumi
endif
ifeq ($(MFEM_USE_HIOP),YES)
   SUBDIRS += hiop
endif
ifeq ($(MFEM_USE_GINKGO),YES)
   SUBDIRS += ginkgo
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
ifeq ($(MFEM_USE_MPI),YES)
ex18p: $(SRC)ex18.hpp
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

# Testing: Specific execution options
ex1-test-seq: ex1
	@$(call mfem-test,$<,, Serial example)
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example)
ex10-test-seq: ex10
	@$(call mfem-test,$<,, Serial example,-tf 5)
ex10p-test-par: ex10p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-tf 5)
ex15-test-seq: ex15
	@$(call mfem-test,$<,, Serial example,-e 1)
ex15p-test-par: ex15p
	@$(call mfem-test,$<, $(RUN_MPI), Parallel example,-e 1)
# Testing: optional tests
ifeq ($(MFEM_USE_STRUMPACK),YES)
ex11p-test-strumpack: ex11p
	@$(call mfem-test,$<, $(RUN_MPI), STRUMPACK example,--strumpack)
test-par-YES: ex11p-test-strumpack
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
	@rm -f refined.mesh displaced.mesh mesh.* ex5.mesh
	@rm -rf Example5* Example9* Example15* Example16* PVExample*
	@rm -f sphere_refined.* sol.* sol_u.* sol_p.* sol_r.* sol_i.*
	@rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
	@rm -f deformed.* velocity.* elastic_energy.* mode_*
	@rm -f ex16.mesh ex16-mesh.* ex16-init.* ex16-final.*
	@rm -f vortex-mesh.* vortex.mesh vortex-?-init.* vortex-?-final.*
	@rm -f deformation.* pressure.*
	@rm -f ex20.dat ex20p_?????.dat gnuplot_ex20.inp gnuplot_ex20p.inp
	@rm -f ex21*.mesh ex21*.sol ex21p_*.*
