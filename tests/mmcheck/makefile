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

# The current MFEM version as an integer, see also `CMakeLists.txt`.
.PHONY: clean

THIS_MK := $(lastword $(MAKEFILE_LIST))
THIS_DIR ?= $(patsubst %/,%,$(dir $(THIS_MK)))
MFEM_DIR := $(THIS_DIR)/../..

# BACKTRACE configuration
BT_DIR = $(realpath $(MFEM_DIR)/../libbacktrace)
BT_INC = -I$(BT_DIR)/include
BT_LIB = -Wl,-rpath,$(BT_DIR)/lib -L$(BT_DIR)/lib -lbacktrace

mmcheck.so: mmcheck.cpp
	$(CXX) -O2 -Wall -shared $(BT_INC) -fPIC -o $@ $< -ldl $(BT_LIB)

clean:
	rm -f mmcheck.*o
