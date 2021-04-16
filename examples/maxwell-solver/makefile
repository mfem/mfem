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
SRC = $(if $(MFEM_DIR:../../..=),$(MFEM_DIR)/examples/maxwell-solver/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_EXAMPLES = helmholtz maxwell maxwell-impedanceBC annulus pml_torus torus_generate_ovlp_partitioning
PAR_EXAMPLES = helmholtzp maxwellp
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean
.PRECIOUS: %.o

COMMON_O= common/PML.o common/MeshPartition.o \
			 common/Utilities.o common/complex_linalg.o common/complex_coeff.o\
			 DST/DST.o ParDST/ParDST.o ParDST/DofMapsDST.o 

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(EXAMPLES)

# Rules for building the EXAMPLES

%: $(SRC)%.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(COMMON_O) $(MFEM_LIBS)

# Rules for compiling miniapp dependencies
$(COMMON_O) $($(EXAMPLES)): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -f DST/*.o
	rm -f ParDST/*.o
	rm -f common/*.o
	rm -f DST2D/*.o
	rm -rf *.dSYM *.TVD.*breakpoints
	rm output/*


