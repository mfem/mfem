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
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
SRC = $(MFEM_DIR)/miniapps/electromagnetics/maglev/
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
TEST_MK = $(MFEM_DIR)/config/test.mk
COMMON_O=../../common/pfem_extras.o

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SEQ_MINIAPPS = maglev
PAR_MINIAPPS = 
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

all: $(MINIAPPS)

# Rules for building the miniapps
$(MINIAPPS): \
%: $(SRC)%.cpp $(COMMON_O) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ ../../common/pfem_extras.o $(MFEM_LIBS)

$(COMMON_O): \
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)

MFEM_TESTS = MINIAPPS
include $(TEST_MK)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	rm -rf maglev
	rm -rf *.000000 *.00000 maglev_000000 maglev_000000.mfem_root
