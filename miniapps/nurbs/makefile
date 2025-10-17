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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/nurbs/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = nurbs_ex1 nurbs_patch_ex1 nurbs_ex3 nurbs_ex5 nurbs_ex24 \
   nurbs_curveint nurbs_printfunc nurbs_solenoidal nurbs_naca_cmesh nurbs_surface
PAR_MINIAPPS = nurbs_ex1p nurbs_ex11p
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(MINIAPPS)

# For out-of-source builds, link the meshes directory from the source tree to
# the build tree. This is needed for running the tests and the sample runs when
# building out-of-source:
ifneq ($(SRC),)
meshes: $(SRC)meshes
	ln -sf $(<) .
$(MINIAPPS): | meshes
.PHONY: copy-data
copy-data: | meshes
endif

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: Parallel vs. serial runs
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp)
%-test-seq: %
	@$(call mfem-test,$<,, NURBS miniapp)

# Additional tests
EX1_ARGS_2 := -r 0 -o 4
EX1_ARGS_3 := -r 2
EX1_ARGS_4 := -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2
EX1_ARGS_5 := -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 0
EX1_ARGS_6 := -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp -r 2
EX1_ARGS_7 := -m ../../data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 0
EX1_ARGS_8 := -m ../../data/pipe-nurbs-2d.mesh -o 2 --weak-bc -r 2
EX1_ARGS_9 := -m ../../data/ball-nurbs.mesh -o 2 --weak-bc -r 0
EX1_ARGS_10 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0
EX1_ARGS_11 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1
EX1_ARGS_12 := -m meshes/two-squares-nurbs.mesh -o 1 -rf meshes/two-squares.ref
EX1_ARGS_13 := -m meshes/two-squares-nurbs-rot.mesh -o 1 -rf meshes/two-squares.ref
EX1_ARGS_14 := -m meshes/two-squares-nurbs-autoedge.mesh -o 1 -rf meshes/two-squares.ref
EX1_ARGS_15 := -m meshes/two-cubes-nurbs.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
EX1_ARGS_16 := -m meshes/two-cubes-nurbs-rot.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
EX1_ARGS_17 := -m meshes/two-cubes-nurbs-autoedge.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
EX1_ARGS_18 := -m meshes/cube-nurbs.mesh -pm "1" -ps "2" -rf meshes/cube.ref

nurbs_ex1-test-seq: nurbs_ex1
	@$(call mfem-test,$<,, NURBS miniapp)
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_2))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_3))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_4))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_5))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_6))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_7))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_8))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_9))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_10))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_11))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_12))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_13))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_14))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_15))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_16))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_17))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1_ARGS_18))

EX1PATCH_ARGS_1 := -incdeg 3 -ref 1 -iro 8 -patcha
EX1PATCH_ARGS_2 := -incdeg 3 -ref 1 -iro 8 -patcha -pa
EX1PATCH_ARGS_3 := -incdeg 3 -ref 1 -iro 8 -patcha -fint
nurbs_patch_ex1-test-seq: nurbs_patch_ex1
ifeq ($(MFEM_USE_LAPACK),YES)
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1PATCH_ARGS_1))
endif
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1PATCH_ARGS_2))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX1PATCH_ARGS_3))

EX3_ARGS_1 := -m $(MFEM_DIR)/data/square-nurbs.mesh -r 1 -o 2
EX3_ARGS_2 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2
nurbs_ex3-test-seq: nurbs_ex3
	@$(call mfem-test,$<,, NURBS miniapp,$(EX3_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX3_ARGS_2))

EX5_ARGS_1 := -m $(MFEM_DIR)/data/square-nurbs.mesh -r 1 -o 2
EX5_ARGS_2 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2
nurbs_ex5-test-seq: nurbs_ex5
	@$(call mfem-test,$<,, NURBS miniapp,$(EX5_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX5_ARGS_2))

EX24_ARGS_1 := -m $(MFEM_DIR)/data/pipe-nurbs-2d.mesh -r 1 -o 2 -p 0
EX24_ARGS_2 := -m $(MFEM_DIR)/data/pipe-nurbs-2d.mesh -r 1 -o 2 -p 2
EX24_ARGS_3 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2 -p 0
EX24_ARGS_4 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2 -p 1
EX24_ARGS_5 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2 -p 2
nurbs_ex24-test-seq: nurbs_ex24
	@$(call mfem-test,$<,, NURBS miniapp,$(EX24_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX24_ARGS_2))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX24_ARGS_3))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX24_ARGS_4))
	@$(call mfem-test,$<,, NURBS miniapp,$(EX24_ARGS_5))

SOL_ARGS_1 := -m $(MFEM_DIR)/data/pipe-nurbs-2d.mesh -r 1 -o 2
SOL_ARGS_2 := -m $(MFEM_DIR)/data/cube-nurbs.mesh -r 1 -o 2
nurbs_solenoidal-test-seq: nurbs_solenoidal
	@$(call mfem-test,$<,, NURBS miniapp,$(SOL_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(SOL_ARGS_2))

CI_ARGS_1 := -uw -n 9 -no-visit
CI_ARGS_2 := -nw -n 9 -no-visit

nurbs_curveint-test-seq: nurbs_curveint
	@$(call mfem-test,$<,, NURBS miniapp,$(CI_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(CI_ARGS_2))

NCT_ARGS_1 := -no-visit -ntail 80 -nbnd 80 -ntip 20 -nwake 40 -sw 2.0 -sbnd 2.5 -stip 1.1 -aoa 3

nurbs_naca_cmesh-test-seq: nurbs_naca_cmesh
	@$(call mfem-test,$<,, NURBS miniapp,$(NCT_ARGS_1))

nurbs_printfunc-test-seq: nurbs_printfunc
	@$(call mfem-test,$<,, NURBS miniapp)

SURF_ARGS_1 := -o 3 -nx 10 -ny 10 -fnx 10 -fny 10 -ex 1 -orig
SURF_ARGS_2 := -o 3 -nx 10 -ny 10 -fnx 40 -fny 40 -ex 1

nurbs_surface-test-seq: nurbs_surface
	@$(call mfem-test,$<,, NURBS miniapp,$(SURF_ARGS_1))
	@$(call mfem-test,$<,, NURBS miniapp,$(SURF_ARGS_2))

EX1P_ARGS_1 :=
EX1P_ARGS_2 := -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp
EX1P_ARGS_3 := -m ../../data/ball-nurbs.mesh -o 2 --weak-bc -r 0
EX1P_ARGS_4 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 0
EX1P_ARGS_5 := -m ../../data/square-disc-nurbs-patch.mesh -o 2 --weak-bc -r 1
EX1P_ARGS_6 := -m meshes/square-nurbs.mesh -r 4  -pm "1" -ps "2"

nurbs_ex1p-test-par: nurbs_ex1p
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_1))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_2))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_3))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_4))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_5))
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX1P_ARGS_6))

EX11P_ARGS_1 :=

nurbs_ex11p-test-par: nurbs_ex11p
	@$(call mfem-test,$<, $(RUN_MPI), NURBS miniapp,$(EX11P_ARGS_1))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh sin-fit.mesh ex5.mesh exsol.mesh mesh.* sol.* mode_*
	@rm -f naca-cmesh.mesh sol_?.gf *-Surface.mesh
	@rm -rf Example1* Example3* Example5* Solenoidal_* ParaView
	@rm -rf CurveInt Naca_cmesh glvis_naca-cmesh.mesh solution.dat
