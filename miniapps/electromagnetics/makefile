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

SEQ_MINIAPPS =
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(SEQ_MINIAPPS) volta tesla
endif

.PHONY: all clean style

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) ../common/pfem_extras.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ ../common/pfem_extras.o $(MFEM_LIBS)

all: $(MINIAPPS)

volta_solver.o: volta_solver.cpp volta_solver.hpp ../common/pfem_extras.hpp \
	$(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -c $(MFEM_FLAGS) volta_solver.cpp -o $@ $(MFEM_LIBS)

tesla_solver.o: tesla_solver.cpp tesla_solver.hpp ../common/pfem_extras.hpp \
	$(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) -c $(MFEM_FLAGS) tesla_solver.cpp -o $@ $(MFEM_LIBS)

volta: volta.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) volta_solver.o ../common/pfem_extras.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ volta_solver.o ../common/pfem_extras.o $(MFEM_LIBS)

tesla: tesla.cpp $(CONFIG_MK) $(MFEM_LIB_FILE) tesla_solver.o ../common/pfem_extras.o
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ tesla_solver.o ../common/pfem_extras.o $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ volta tesla
	rm -rf Volta-AMR* Tesla-AMR*
	rm -rf *.dSYM *.TVD.*breakpoints
