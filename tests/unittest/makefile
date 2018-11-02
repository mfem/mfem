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

MFEM_DIR = ../../
CONFIG_MK = $(MFEM_DIR)/config/config.mk
USER_CONFIG = $(MFEM_DIR)/config/user.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)
-include $(USER_MK)

CC = $(MFEM_CXX)
CCOPTS     = -g
CCC = $(CC) $(CCOPTS)

INCLUDES = -I. $(MFEM_FLAGS)

DIRS = 	fem linalg mesh
SOURCE_FILES = unit_test_main.cpp $(sort $(wildcard ./*/*.cpp))
HEADER_FILES = catch.hpp
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

.SUFFIXES: .cpp .o
.cpp.o:
	$(CCC) -c $(<) $(INCLUDES) -o $(@)

test: $(OBJECT_FILES) $(MFEM_LIB_FILE)
	$(CCC) $(OBJECT_FILES) $(INCLUDES) $(MFEM_LIBS) $(EXT_LIBDIRS) $(EXT_LIBS) -o test

$(OBJECT_FILES): $(HEADER_FILES)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o */*.o */*~ *~

distclean: clean
	rm -rf test test.dSYM output_meshes

ASTYLE = astyle --options=$(MFEM_DIR)/config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS),"$(dir)/*.?pp")

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
