# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Serial compiler
CC         = g++
CCOPTS     = -g
CCC = $(CC) $(CCOPTS)

MFEM_DIR = ../../mfem

COVERAGE ?= NO
COVLIBS = 
ifeq ($(COVERAGE),YES)
   COVLIBS = -lgcov
endif

INCLUDES = -I. -I$(MFEM_DIR) -I$(MFEM_DIR)/fem -I$(MFEM_DIR)/general -I$(MFEM_DIR)/linalg -I$(MFEM_DIR)/mesh
LIB_FILES = $(MFEM_DIR)/libmfem.a

SOURCE_FILES = unit_test_main.cpp $(sort $(wildcard ./*/*.cpp))
HEADER_FILES = catch.hpp
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

.SUFFIXES: .cpp .o
.cpp.o:
	$(CCC) -c $(<D)/$(<F) $(INCLUDES) -o $(<D)/$(<F:.cpp=.o)

test: $(OBJECT_FILES) $(LIB_FILES)
	$(CCC) $(OBJECT_FILES) $(INCLUDES) -L$(MFEM_DIR) $(EXT_LIBDIRS) -lmfem $(COVLIBS) $(EXT_LIBS) -o test

$(OBJECT_FILES): $(HEADER_FILES)

clean:
	rm -f *.o */*.o */*~ *~

distclean: clean
	rm -f test
