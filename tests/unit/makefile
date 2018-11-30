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

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/tests/unit/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

CC = $(MFEM_CXX)
CCOPTS = -g
CCC = $(CC) $(CCOPTS)

# -I$(MFEM_DIR) is needed by some tests, e.g. to #include "general/text.hpp"
INCLUDES = $(MFEM_FLAGS) -I$(or $(SRC:%/=%),.) -I$(MFEM_DIR)

SOURCE_FILES = $(SRC)unit_test_main.cpp $(sort $(wildcard $(SRC)*/*.cpp))
HEADER_FILES = $(SRC)catch.hpp
OBJECT_FILES = $(SOURCE_FILES:$(SRC)%.cpp=%.o)
DATA_DIR = data

.SUFFIXES: .cpp .o

test: $(OBJECT_FILES) $(MFEM_LIB_FILE) $(CONFIG_MK) $(DATA_DIR)
	$(CCC) $(OBJECT_FILES) $(INCLUDES) $(MFEM_LIBS) -o $(@)

$(OBJECT_FILES): %.o: $(SRC)%.cpp $(HEADER_FILES) $(CONFIG_MK)
	@mkdir -p $(@D)
	$(CCC) -c $(<) $(INCLUDES) -o $(@)

$(DATA_DIR):
	ln -s $(SRC)$(DATA_DIR) .

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f test *.o */*.o */*~ *~
	rm -rf test.dSYM output_meshes
