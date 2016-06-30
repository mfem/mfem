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

define MFEM_HELP_MSG

MFEM makefile targets:

   make config
   make
   make all
   make status/info
   make serial
   make parallel
   make debug
   make pdebug
   make check/test
   make install
   make clean
   make distclean
   make style

Examples:

make config MFEM_USE_MPI=YES MFEM_DEBUG=YES MPICXX=mpiCC
   Configure the make system for subsequent runs (analogous to a configure script).
   The available options are documented in the INSTALL file.
make -j 4
   Build the library (in parallel) using the current configuration options.
make all
   Build the library, the examples and the miniapps using the current configuration.
make status
   Display information about the current configuration.
make serial
   A shortcut to configure and build the serial optimized version of the library.
make parallel
   A shortcut to configure and build the parallel optimized version of the library.
make debug
   A shortcut to configure and build the serial debug version of the library.
make pdebug
   A shortcut to configure and build the parallel debug version of the library.
make check
   Quick-check the build by compiling and running Example 1/1p.
make test
   Verify the build by checking the results from running all examples and miniapps.
make install PREFIX=<dir>
   Install the library and headers in <dir>/lib and <dir>/include.
make clean
   Clean the library and object files, but keep configuration.
make distclean
   In addition to "make clean", clean the configuration and remove the local
   installation directory.
make style
   Format the MFEM C++ source files using Artistic Style (astyle).

endef

# Path to the mfem directory relative to the compile directory:
MFEM_DIR = .
# ... or simply an absolute path
# MFEM_DIR = $(realpath .)

CONFIG_MK = config/config.mk

DEFAULTS_MK = config/defaults.mk
include $(DEFAULTS_MK)

# Optional user config file, see config/defaults.mk
USER_CONFIG = config/user.mk
-include $(USER_CONFIG)

# Helper print-info function
mfem-info = $(if $(filter YES,$(VERBOSE)),$(info *** [info]$(1)),)

$(call mfem-info, MAKECMDGOALS = $(MAKECMDGOALS))

# Include $(CONFIG_MK) unless some of the $(SKIP_INCLUDE_TARGETS) are given
SKIP_INCLUDE_TARGETS = help config clean distclean serial parallel debug pdebug\
 style
HAVE_SKIP_INCLUDE_TARGET = $(filter $(SKIP_INCLUDE_TARGETS),$(MAKECMDGOALS))
ifeq (,$(HAVE_SKIP_INCLUDE_TARGET))
   $(call mfem-info, Including $(CONFIG_MK))
   -include $(CONFIG_MK)
else
   # Do not allow skip-include targets to be combined with other targets
   ifneq (1,$(words $(MAKECMDGOALS)))
      $(error Target '$(firstword $(HAVE_SKIP_INCLUDE_TARGET))' can not be\
      combined with other targets)
   endif
   $(call mfem-info, NOT including $(CONFIG_MK))
endif

# Compile flags used by MFEM: CPPFLAGS, CXXFLAGS, plus library flags
INCFLAGS = -I@MFEM_INC_DIR@
# Link flags used by MFEM: library link flags plus LDFLAGS (added last)
ALL_LIBS = -L@MFEM_LIB_DIR@ -lmfem

# The default value of CXXFLAGS is based on the value of MFEM_DEBUG
ifeq ($(MFEM_DEBUG),YES)
   CXXFLAGS ?= $(DEBUG_FLAGS)
endif
CXXFLAGS ?= $(OPTIM_FLAGS)

# MPI configuration
ifneq ($(MFEM_USE_MPI),YES)
   MFEM_CXX ?= $(CXX)
else
   MFEM_CXX ?= $(MPICXX)
   INCFLAGS += $(METIS_OPT) $(HYPRE_OPT)
   ALL_LIBS += $(METIS_LIB) $(HYPRE_LIB)
endif

DEP_CXX ?= $(MFEM_CXX)

# LAPACK library configuration
ifeq ($(MFEM_USE_LAPACK),YES)
   INCFLAGS += $(LAPACK_OPT)
   ALL_LIBS += $(LAPACK_LIB)
endif

# OpenMP configuration
ifeq ($(MFEM_USE_OPENMP),YES)
   MFEM_THREAD_SAFE ?= YES
   ifneq ($(MFEM_THREAD_SAFE),YES)
      $(error Incompatible config: MFEM_USE_OPENMP requires MFEM_THREAD_SAFE)
   endif
   INCFLAGS += $(OPENMP_OPT)
   ALL_LIBS += $(OPENMP_LIB)
endif

ifeq ($(MFEM_TIMER_TYPE),2)
   ALL_LIBS += $(POSIX_CLOCKS_LIB)
endif

# MESQUITE library configuration
ifeq ($(MFEM_USE_MESQUITE),YES)
   INCFLAGS += $(MESQUITE_OPT)
   ALL_LIBS += $(MESQUITE_LIB)
endif

# SuiteSparse library configuration
ifeq ($(MFEM_USE_SUITESPARSE),YES)
   INCFLAGS += $(SUITESPARSE_OPT)
   ALL_LIBS += $(SUITESPARSE_LIB)
endif

# SuperLU library configuration
ifeq ($(MFEM_USE_SUPERLU),YES)
   INCFLAGS += $(SUPERLU_OPT)
   ALL_LIBS += $(SUPERLU_LIB)
endif

# Gecko library configuration
ifeq ($(MFEM_USE_GECKO),YES)
   INCFLAGS += $(GECKO_OPT)
   ALL_LIBS += $(GECKO_LIB)
endif

# GnuTLS library configuration
ifeq ($(MFEM_USE_GNUTLS),YES)
   INCFLAGS += $(GNUTLS_OPT)
   ALL_LIBS += $(GNUTLS_LIB)
endif

# NetCDF library configuration
ifeq ($(MFEM_USE_NETCDF),YES)
   INCFLAGS += $(NETCDF_OPT)
   ALL_LIBS += $(NETCDF_LIB)
endif

# List of all defines that may be enabled in config.hpp and config.mk:
MFEM_DEFINES = MFEM_USE_MPI MFEM_USE_METIS_5 MFEM_DEBUG MFEM_USE_LAPACK\
 MFEM_THREAD_SAFE MFEM_USE_OPENMP MFEM_USE_MEMALLOC MFEM_TIMER_TYPE\
 MFEM_USE_MESQUITE MFEM_USE_SUITESPARSE MFEM_USE_GECKO MFEM_USE_SUPERLU\
 MFEM_USE_GNUTLS MFEM_USE_NETCDF

# List of makefile variables that will be written to config.mk:
MFEM_CONFIG_VARS = MFEM_CXX MFEM_CPPFLAGS MFEM_CXXFLAGS MFEM_INC_DIR\
 MFEM_INCFLAGS MFEM_FLAGS MFEM_LIB_DIR MFEM_LIBS MFEM_LIB_FILE MFEM_BUILD_TAG\
 MFEM_PREFIX

# Config vars: values of the form @VAL@ are replaced by $(VAL) in config.mk
MFEM_CPPFLAGS  ?= $(CPPFLAGS)
MFEM_CXXFLAGS  ?= $(CXXFLAGS)
MFEM_INCFLAGS  ?= $(INCFLAGS)
MFEM_FLAGS     ?= @MFEM_CPPFLAGS@ @MFEM_CXXFLAGS@ @MFEM_INCFLAGS@
MFEM_LIBS      ?= $(ALL_LIBS) $(LDFLAGS)
MFEM_LIB_FILE  ?= @MFEM_LIB_DIR@/libmfem.a
MFEM_BUILD_TAG ?= $(shell uname -snm)
MFEM_PREFIX    ?= $(PREFIX)
MFEM_INC_DIR   ?= @MFEM_DIR@
MFEM_LIB_DIR   ?= @MFEM_DIR@

# If we have 'config' target, export variables used by config/makefile
ifneq (,$(filter config,$(MAKECMDGOALS)))
   export $(MFEM_DEFINES) MFEM_DEFINES $(MFEM_CONFIG_VARS) MFEM_CONFIG_VARS
   export VERBOSE
endif

# If we have 'install' target, export variables used by config/makefile
ifneq (,$(filter install,$(MAKECMDGOALS)))
   ifneq (install,$(MAKECMDGOALS))
      $(error Target 'install' can not be combined with other targets)
   endif
   # Allow changing the PREFIX during install with: make install PREFIX=<dir>
   PREFIX := $(MFEM_PREFIX)
   PREFIX_INC := $(PREFIX)/include
   PREFIX_LIB := $(PREFIX)/lib
   MFEM_PREFIX := $(abspath $(PREFIX))
   MFEM_DIR := $(abspath .)
   MFEM_INC_DIR = $(abspath $(PREFIX_INC))
   MFEM_LIB_DIR = $(abspath $(PREFIX_LIB))
   export $(MFEM_DEFINES) MFEM_DEFINES $(MFEM_CONFIG_VARS) MFEM_CONFIG_VARS
   export VERBOSE
endif

# Source dirs in logical order
DIRS = general linalg mesh fem
SOURCE_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

.PHONY: lib all clean distclean install config status info deps serial parallel\
 debug pdebug style check test

.SUFFIXES: .cpp .o
.cpp.o:
	$(MFEM_CXX) $(MFEM_FLAGS) -c $(<) -o $(@)


lib: libmfem.a

all: lib
	$(MAKE) -C examples
	$(MAKE) -C miniapps/common
	$(MAKE) -C miniapps/meshing
	$(MAKE) -C miniapps/electromagnetics
	$(MAKE) -C miniapps/performance

-include deps.mk

$(OBJECT_FILES): $(CONFIG_MK)

libmfem.a: $(OBJECT_FILES)
	$(AR) $(ARFLAGS) libmfem.a $(OBJECT_FILES)
	$(RANLIB) libmfem.a

serial:
	$(MAKE) config MFEM_USE_MPI=NO MFEM_DEBUG=NO && $(MAKE)

parallel:
	$(MAKE) config MFEM_USE_MPI=YES MFEM_DEBUG=NO && $(MAKE)

debug:
	$(MAKE) config MFEM_USE_MPI=NO MFEM_DEBUG=YES && $(MAKE)

pdebug:
	$(MAKE) config MFEM_USE_MPI=YES MFEM_DEBUG=YES && $(MAKE)

deps:
	rm -f deps.mk
	for i in $(SOURCE_FILES:.cpp=); do \
	   $(DEP_CXX) $(MFEM_FLAGS) -MM -MT $${i}.o $${i}.cpp >> deps.mk; done

check: lib
	@printf "Quick-checking the MFEM library."
	@printf " Use 'make test' for more extensive tests.\n"
	@$(MAKE) -C examples \
	$(if $(findstring YES,$(MFEM_USE_MPI)),ex1p-test-par,ex1-test-seq)

test: lib
	@echo "Testing the MFEM library. This may take a while..."
	@echo "Building all examples and miniapps..."
	@make all
	@echo "Running examples..."
	@$(MAKE) -C examples test
	@echo "Running meshing miniapps..."
	@$(MAKE) -C miniapps/meshing test
	@echo "Running electromagnetic miniapps..."
	@$(MAKE) -C miniapps/electromagnetics test
	@echo "Running high-performance miniapps..."
	@$(MAKE) -C miniapps/performance test
	@echo "Done."

clean:
	rm -f */*.o */*~ *~ libmfem.a deps.mk
	$(MAKE) -C examples clean
	$(MAKE) -C miniapps/common clean
	$(MAKE) -C miniapps/meshing clean
	$(MAKE) -C miniapps/electromagnetics clean
	$(MAKE) -C miniapps/performance clean

distclean: clean
	rm -rf mfem/
	$(MAKE) -C config clean
	$(MAKE) -C doc clean

install: libmfem.a
# install static library
	mkdir -p $(PREFIX_LIB)
	$(INSTALL) -m 640 libmfem.a $(PREFIX_LIB)
# install top level includes
	mkdir -p $(PREFIX_INC)
	$(INSTALL) -m 640 mfem.hpp mfem-performance.hpp $(PREFIX_INC)
# install config include
	mkdir -p $(PREFIX_INC)/config
	$(INSTALL) -m 640 config/config.hpp config/tconfig.hpp $(PREFIX_INC)/config
# install remaining includes in each subdirectory
	for dir in $(DIRS); do \
	   mkdir -p $(PREFIX_INC)/$$dir && \
	   $(INSTALL) -m 640 $$dir/*.hpp $(PREFIX_INC)/$$dir; done
# install config.mk at root of install tree
	$(MAKE) -C config config-mk CONFIG_MK=config-install.mk
	$(INSTALL) -m 640 config/config-install.mk $(PREFIX)/config.mk
	rm -f config/config-install.mk

$(CONFIG_MK):
	$(info )
	$(info MFEM is not configured.)
	$(info Run "make config" first, or see "make help".)
	$(info )
	$(error )

config:
	$(MAKE) -C config all

help:
	$(info $(value MFEM_HELP_MSG))
	@true

status info:
	$(info MFEM_USE_MPI         = $(MFEM_USE_MPI))
	$(info MFEM_USE_METIS_5     = $(MFEM_USE_METIS_5))
	$(info MFEM_DEBUG           = $(MFEM_DEBUG))
	$(info MFEM_USE_LAPACK      = $(MFEM_USE_LAPACK))
	$(info MFEM_THREAD_SAFE     = $(MFEM_THREAD_SAFE))
	$(info MFEM_USE_OPENMP      = $(MFEM_USE_OPENMP))
	$(info MFEM_USE_MEMALLOC    = $(MFEM_USE_MEMALLOC))
	$(info MFEM_TIMER_TYPE      = $(MFEM_TIMER_TYPE))
	$(info MFEM_USE_MESQUITE    = $(MFEM_USE_MESQUITE))
	$(info MFEM_USE_SUITESPARSE = $(MFEM_USE_SUITESPARSE))
	$(info MFEM_USE_SUPERLU     = $(MFEM_USE_SUPERLU))
	$(info MFEM_USE_GECKO       = $(MFEM_USE_GECKO))
	$(info MFEM_USE_GNUTLS      = $(MFEM_USE_GNUTLS))
	$(info MFEM_USE_NETCDF      = $(MFEM_USE_NETCDF))
	$(info MFEM_CXX             = $(value MFEM_CXX))
	$(info MFEM_CPPFLAGS        = $(value MFEM_CPPFLAGS))
	$(info MFEM_CXXFLAGS        = $(value MFEM_CXXFLAGS))
	$(info MFEM_INCFLAGS        = $(value MFEM_INCFLAGS))
	$(info MFEM_FLAGS           = $(value MFEM_FLAGS))
	$(info MFEM_LIBS            = $(value MFEM_LIBS))
	$(info MFEM_LIB_FILE        = $(value MFEM_LIB_FILE))
	$(info MFEM_BUILD_TAG       = $(value MFEM_BUILD_TAG))
	$(info MFEM_PREFIX          = $(value MFEM_PREFIX))
	$(info MFEM_INC_DIR         = $(value MFEM_INC_DIR))
	$(info MFEM_LIB_DIR         = $(value MFEM_LIB_DIR))
	@true

ASTYLE = astyle --options=config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS) examples $(wildcard miniapps/*),"$(dir)/*.?pp")

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi
