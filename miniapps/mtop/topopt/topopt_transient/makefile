# Makefile for topology optimization with transient dynamics

MFEM_DIR ?= ../../../..
MFEM_BUILD_DIR ?= ../../../..
MFEM_INSTALL_DIR ?= ../../../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/mtop/topopt/topopt_transient/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Shared source compiled into local objects
COMMON_OBJ = MMA_MFEM.o LatentMirrorOptimizer.o mtop_solvers.o pde_filter.o

SEQ_MINIAPPS =
PAR_MINIAPPS = TopOptTransient test_convergence_study test_physical_validation ElastTopOpt_transient ForwardElastodynamics test_unified_solver test_gradient_taylor SemiInfiniteWave SemiInfiniteWave_damped SemiInfiniteWave_proper_damping SemiInfiniteWave_checkpoint_ready SemiInfiniteWave_gmsh SemiInfiniteWave_semicircle SimpleWaveTest

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

# TopOptTransient needs MMA
TopOptTransient: TopOptTransient.o MMA_MFEM.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# MMA optimizer from ../../mma/
MMA_MFEM.o: $(SRC)../../mma/MMA_MFEM.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

LatentMirrorOptimizer.o: $(SRC)../../mma/LatentMirrorOptimizer.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# mtop solvers from ../../
mtop_solvers.o: $(SRC)../../mtop_solvers.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

pde_filter.o: $(SRC)../../pde_filter.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf ParaView
