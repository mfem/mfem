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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/drl4amr/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# Some choices below are based on the OS type:
NOTMAC := $(subst Darwin,,$(shell uname -s))
ifneq ($(NOTMAC),)
   SOPREFIX = so
else
   SOPREFIX = install_
endif

SEQ_MINIOBJS = drl4amr.o
PAR_MINIOBJS =
ifeq ($(MFEM_USE_MPI),NO)
   MINIOBJS = $(SEQ_MINIOBJS)
else
   MINIOBJS = $(PAR_MINIOBJS) $(SEQ_MINIOBJS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all cln clean clean-build clean-exec

# Remove built-in rule
%.o: %.cpp

# Replace the default implicit rule for *.cpp files
%.o: $(SRC)%.cpp $(SRC)%.hpp $(CONFIG_MK)
	$(MFEM_CXX) -c -fPIC $(MFEM_FLAGS) $< -o $@

all: libdrl4amr.so drl4amr

go:;./drl4amr
py python:;python drl4amr.py
tst test:;$(MAKE) go && $(MAKE) python

drl4amr: main.cpp libdrl4amr.so
	$(MFEM_CXX) $(MFEM_FLAGS) main.cpp -o drl4amr -L. -Wl,-rpath,. -ldrl4amr

libdrl4amr.so: $(MINIOBJS)
	$(MFEM_CXX) -shared -Wl,-$(SOPREFIX)name,libdrl4amr.so -o libdrl4amr.so $(MINIOBJS) $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

cln clean:
	rm -f *.o *~
	rm -rf *.dSYM *.TVD.*breakpoints $(SEQ_MINIOBJS) drl4amr libdrl4amr.so
