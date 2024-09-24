# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/petsc/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES =
PAR_EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex6p ex9p ex10p
ifeq ($(MFEM_USE_SLEPC),YES)
   PAR_EXAMPLES += ex11p
endif
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES)
endif
RC_FILES = $(patsubst $(SRC)%,%,$(wildcard $(SRC)rc_*))

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

# Examples depend on their corresponding rc_* files:
make-rc-rule = $(1): | $(filter rc_$(1)%,$(RC_FILES))
$(foreach ex,$(EXAMPLES),$(eval $(call make-rc-rule,$(ex))))

# Rules to copy the rc_* files when building out-of-source:
ifneq ($(SRC),)
$(RC_FILES): %: $(SRC)%
	cp -pf $(<) .
endif

ifeq ($(MFEM_USE_PETSC),NO)
$(EXAMPLES):
	$(error MFEM is not configured with PETSC)
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

# Testing: Parallel runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
TESTNAME      = Parallel PETSc example
TESTNAME_CUDA = Parallel CUDA PETSc example
TESTNAME_HIP  = Parallel HIP PETSc example
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME))


# Testing PETSc execution options.
EX1_ARGS_W            := -m ../../data/amr-quad.mesh --usepetsc
EX1_ARGS_P            := -m ../../data/amr-quad.mesh --usepetsc --petscopts rc_ex1p
EX1_ARGS_CUDA         := -m ../../data/star.mesh --usepetsc --partial-assembly --device cuda --petscopts rc_ex1p_device
EX1_ARGS_CUDAAMG      := -m ../../data/star.mesh --usepetsc --device cuda --petscopts rc_ex1p_deviceamg
EX1_ARGS_HIP          := -m ../../data/star.mesh --usepetsc --partial-assembly --device hip --petscopts rc_ex1p_device
EX1_ARGS_HIPAMG       := -m ../../data/star.mesh --usepetsc --device hip --petscopts rc_ex1p_deviceamg
EX2_ARGS              := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex2p
EX2_ARGS_BDDC         := -m ../../data/beam-tri.mesh --usepetsc --nonoverlapping --petscopts rc_ex2p_bddc
EX2_ARGS_ASM          := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex2p_asm
EX3_ARGS_BDDC_2D      := -m ../../data/klein-bottle.mesh -o 2 -f 0.1 --usepetsc --petscopts rc_ex3p_bddc --nonoverlapping
EX3_ARGS_BDDC_3D      := -m ../../data/amr-hex.mesh -rs 1 -rp 0 -o 2 -f 0.1 --usepetsc --petscopts rc_ex3p_bddc --nonoverlapping
EX4_ARGS              := -m ../../data/klein-bottle.mesh -o 2 --usepetsc --petscopts rc_ex4p_bddc --nonoverlapping
EX4_HYB_ARGS          := -m ../../data/klein-bottle.mesh -o 2 --usepetsc --petscopts rc_ex4p_bddc --nonoverlapping --hybridization
EX5_BDDC_LB_ARGS      := -m ../../data/star.mesh --usepetsc -o 0 --petscopts rc_ex5p_bddc --nonoverlapping --local-bdr
EX5_BDDC_GB_ARGS      := -m ../../data/star.mesh --usepetsc -o 0 --petscopts rc_ex5p_bddc --nonoverlapping
EX5_FSPL_ARGS         := -m ../../data/beam-tet.mesh --usepetsc -o 0 --petscopts rc_ex5p_fieldsplit
EX6_ARGS              := -m ../../data/amr-quad.mesh --usepetsc
EX6_NONOVL_ARGS       := -m ../../data/amr-quad.mesh --usepetsc --nonoverlapping
EX9_E_ARGS            := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl -dt 0.1
EX9_ES_ARGS           := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl --no-step
EX9_ES_ARGS_CUDA      := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl_device --no-step --partial-assembly --device cuda
EX9_ES_ARGS_HIP       := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_expl_device --no-step --partial-assembly --device hip
EX9_IS_ARGS           := -m ../../data/periodic-hexagon.mesh --usepetsc --petscopts rc_ex9p_impl --implicit -tf 0.5
EX10_ARGS             := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex10p -tf 30 -s 3 -rs 2 -dt 3
EX10_MF_ARGS          := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex10p_mf -tf 6 -s 3 -rs 0 -dt 3
EX10_MFOP_ARGS        := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex10p_mfop -tf 6 -s 3 -rs 0 -dt 3
EX10_JFNK_ARGS        := -m ../../data/beam-quad.mesh --usepetsc --petscopts rc_ex10p_jfnk --jfnk -tf 6 -s 3 -rs 0 -dt 3
EX11_ARGS_SINV        := -m ../../data/star.mesh --useslepc
EX11_ARGS_LOBPCG      := -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_lobpcg
EX11_ARGS_LOBPCG_CUDA := -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_lobpcg_device --device cuda
EX11_ARGS_LOBPCG_HIP  := -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_lobpcg_device --device hip
EX11_ARGS_GD          := -m ../../data/star.mesh --useslepc --slepcopts rc_ex11p_gd

ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX1_ARGS_W))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX1_ARGS_P))
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX1_ARGS_CUDA))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX1_ARGS_CUDAAMG))
endif
ifeq ($(MFEM_USE_HIP),YES)
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_HIP),$(EX1_ARGS_HIP))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_HIP),$(EX1_ARGS_HIPAMG))
endif
ex2p-test-par: ex2p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX2_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX2_ARGS_BDDC))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX2_ARGS_ASM))
ex3p-test-par: ex3p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX3_ARGS_BDDC_2D))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX3_ARGS_BDDC_3D))
ex4p-test-par: ex4p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX4_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX4_HYB_ARGS))
ex5p-test-par: ex5p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX5_BDDC_LB_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX5_BDDC_GB_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX5_FSPL_ARGS))
ex6p-test-par: ex6p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX6_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX6_NONOVL_ARGS))
ex9p-test-par: ex9p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX9_E_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX9_ES_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX9_IS_ARGS))
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX9_ES_ARGS_CUDA))
endif
ifeq ($(MFEM_USE_HIP),YES)
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_HIP),$(EX9_ES_ARGS_HIP))
endif
ex10p-test-par: ex10p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX10_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX10_MF_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX10_MFOP_ARGS))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX10_JFNK_ARGS))
ifeq ($(MFEM_USE_SLEPC),YES)
ex11p-test-par: ex11p
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX11_ARGS_SINV))
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX11_ARGS_LOBPCG))
ifeq ($(MFEM_USE_CUDA),YES)
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_CUDA),$(EX11_ARGS_LOBPCG_CUDA))
endif
# SLEPc does not support BVSVEC with HIP
#ifeq ($(MFEM_USE_HIP),YES)
#	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME_HIP),$(EX11_ARGS_LOBPCG_HIP))
#endif
	@$(call mfem-test,$<, $(RUN_MPI), $(TESTNAME),$(EX11_ARGS_GD))
endif

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints
ifneq ($(SRC),)
	rm -f $(RC_FILES)
endif

clean-exec:
	@rm -rf mesh.* sol.* sol_p.* sol_u.* Example5*
	@rm -f ex9-mesh.* ex9-init.* ex9-final.* Example9*
	@rm -f deformed.* velocity.* elastic_energy.* mode_*
