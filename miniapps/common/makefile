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
MFEM_DIR = ../..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_DIR = ../../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

SEQ_MINIOBJS = fem_extras.o
ifeq ($(MFEM_USE_MPI),NO)
   MINIOBJS = $(SEQ_MINIOBJS)
else
   MINIOBJS = $(SEQ_MINIOBJS) pfem_extras.o
endif

.PHONY: all clean style

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
all: $(MINIOBJS)

fem_extras.o: fem_extras.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -c $(MFEM_FLAGS) fem_extras.cpp -o $@ $(MFEM_LIBS)

pfem_extras.o: pfem_extras.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -c $(MFEM_FLAGS) pfem_extras.cpp -o $@ $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~
	rm -rf *.dSYM *.TVD.*breakpoints
