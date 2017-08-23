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
MFEM_VERSION = 30301
MFEM_VERSION_STRING = $(shell printf "%06d" $(MFEM_VERSION) | \
  sed -e 's/^0*\(.*.\)\(..\)\(..\)$$/\1.\2.\3/' -e 's/\.0/./g' -e 's/\.0$$//')

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
make config BUILD_DIR=<dir>
   Configure an out-of-source-tree build in the given directory.
make config -f <mfem-dir>/makefile
   Configure an out-of-source-tree build in the current directory.
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

# Path to the mfem source directory, defaults to this makefile's directory:
THIS_MK := $(lastword $(MAKEFILE_LIST))
$(if $(wildcard $(THIS_MK)),,$(error Makefile not found "$(THIS_MK)"))
MFEM_DIR ?= $(patsubst %/,%,$(dir $(THIS_MK)))
MFEM_REAL_DIR := $(realpath $(MFEM_DIR))
$(if $(MFEM_REAL_DIR),,$(error Source directory "$(MFEM_DIR)" is not valid))
SRC := $(if $(MFEM_REAL_DIR:$(CURDIR)=),$(MFEM_DIR)/,)
$(if $(word 2,$(SRC)),$(error Spaces in SRC = "$(SRC)" are not supported))

EXAMPLE_SUBDIRS = sundials petsc
EXAMPLE_DIRS := examples $(addprefix examples/,$(EXAMPLE_SUBDIRS))
EXAMPLE_TEST_DIRS := examples

MINIAPP_SUBDIRS = common electromagnetics meshing performance tools
MINIAPP_DIRS := $(addprefix miniapps/,$(MINIAPP_SUBDIRS))
MINIAPP_TEST_DIRS := $(filter-out %/common,$(MINIAPP_DIRS))
MINIAPP_USE_COMMON := $(addprefix miniapps/,electromagnetics tools)

EM_DIRS = $(EXAMPLE_DIRS) $(MINIAPP_DIRS)
EM_TEST_DIRS = $(EXAMPLE_TEST_DIRS) $(MINIAPP_TEST_DIRS)

# Use BUILD_DIR on the command line; set MFEM_BUILD_DIR before including this
# makefile or config/config.mk from a separate $(BUILD_DIR).
MFEM_BUILD_DIR ?= .
BUILD_DIR := $(MFEM_BUILD_DIR)
BUILD_REAL_DIR := $(abspath $(BUILD_DIR))
ifneq ($(BUILD_REAL_DIR),$(MFEM_REAL_DIR))
   BUILD_SUBDIRS = $(DIRS) config $(EM_DIRS) doc
   BUILD_DIR_DEF = -DMFEM_BUILD_DIR="$(BUILD_REAL_DIR)"
   BLD := $(if $(BUILD_REAL_DIR:$(CURDIR)=),$(BUILD_DIR)/,)
   $(if $(word 2,$(BLD)),$(error Spaces in BLD = "$(BLD)" are not supported))
else
   BUILD_DIR = $(MFEM_DIR)
   BLD := $(SRC)
endif
MFEM_BUILD_DIR := $(BUILD_DIR)

CONFIG_MK = $(BLD)config/config.mk

DEFAULTS_MK = $(SRC)config/defaults.mk
include $(DEFAULTS_MK)

# Optional user config file, see config/defaults.mk
USER_CONFIG = $(BLD)config/user.mk
-include $(USER_CONFIG)

# Helper print-info function
mfem-info = $(if $(filter YES,$(VERBOSE)),$(info *** [info]$(1)),)
export VERBOSE

$(call mfem-info, MAKECMDGOALS = $(MAKECMDGOALS))
$(call mfem-info, MAKEFLAGS    = $(MAKEFLAGS))
$(call mfem-info, MFEM_DIR  = $(MFEM_DIR))
$(call mfem-info, BUILD_DIR = $(BUILD_DIR))
$(call mfem-info, SRC       = $(SRC))
$(call mfem-info, BLD       = $(BLD))

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
INCFLAGS =
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
   $(foreach mpidep,SUPERLU PETSC,$(if $(MFEM_USE_$(mpidep):NO=),\
     $(warning *** [MPI is OFF] setting MFEM_USE_$(mpidep) = NO)\
     $(eval override MFEM_USE_$(mpidep)=NO),))
else
   MFEM_CXX ?= $(MPICXX)
   INCFLAGS += $(METIS_OPT) $(HYPRE_OPT)
   ALL_LIBS += $(METIS_LIB) $(HYPRE_LIB)
endif

DEP_CXX ?= $(MFEM_CXX)

# Check OpenMP configuration
ifeq ($(MFEM_USE_OPENMP),YES)
   MFEM_THREAD_SAFE ?= YES
   ifneq ($(MFEM_THREAD_SAFE),YES)
      $(error Incompatible config: MFEM_USE_OPENMP requires MFEM_THREAD_SAFE)
   endif
endif

# Check STRUMPACK configuration
ifeq ($(MFEM_USE_STRUMPACK),YES)
   MFEM_USE_OPENMP ?= YES
   ifneq ($(MFEM_USE_OPENMP),YES)
      $(error Incompatible config: MFEM_USE_STRUMPACK requires MFEM_USE_OPENMP)
   endif
endif

# List of MFEM dependencies, processed below
MFEM_DEPENDENCIES = LIBUNWIND SIDRE LAPACK OPENMP SUNDIALS MESQUITE SUITESPARSE\
 SUPERLU STRUMPACK GECKO GNUTLS NETCDF PETSC MPFR

# Macro for adding dependencies
define mfem_add_dependency
ifeq ($(MFEM_USE_$(1)),YES)
   INCFLAGS += $($(1)_OPT)
   ALL_LIBS += $($(1)_LIB)
endif
endef

# Process dependencies
$(foreach dep,$(MFEM_DEPENDENCIES),$(eval $(call mfem_add_dependency,$(dep))))

# Timer option
ifeq ($(MFEM_TIMER_TYPE),2)
   ALL_LIBS += $(POSIX_CLOCKS_LIB)
endif

# gzstream configuration
ifeq ($(MFEM_USE_GZSTREAM),YES)
   ALL_LIBS += -lz
endif

# List of all defines that may be enabled in config.hpp and config.mk:
MFEM_DEFINES = MFEM_VERSION MFEM_USE_MPI MFEM_NO_METIS MFEM_USE_METIS_5\
 MFEM_DEBUG MFEM_USE_GZSTREAM MFEM_USE_LIBUNWIND MFEM_USE_LAPACK\
 MFEM_THREAD_SAFE MFEM_USE_OPENMP MFEM_USE_MEMALLOC MFEM_TIMER_TYPE\
 MFEM_USE_SUNDIALS MFEM_USE_MESQUITE MFEM_USE_SUITESPARSE MFEM_USE_GECKO\
 MFEM_USE_SUPERLU MFEM_USE_STRUMPACK MFEM_USE_GNUTLS MFEM_USE_NETCDF\
 MFEM_USE_PETSC MFEM_USE_MPFR MFEM_USE_SIDRE

# List of makefile variables that will be written to config.mk:
MFEM_CONFIG_VARS = MFEM_CXX MFEM_CPPFLAGS MFEM_CXXFLAGS MFEM_INC_DIR\
 MFEM_TPLFLAGS MFEM_INCFLAGS MFEM_FLAGS MFEM_LIB_DIR MFEM_LIBS MFEM_LIB_FILE\
 MFEM_BUILD_TAG MFEM_PREFIX MFEM_CONFIG_EXTRA MFEM_MPIEXEC MFEM_MPIEXEC_NP

# Config vars: values of the form @VAL@ are replaced by $(VAL) in config.mk
MFEM_CPPFLAGS  ?= $(CPPFLAGS)
MFEM_CXXFLAGS  ?= $(CXXFLAGS)
MFEM_TPLFLAGS  ?= $(INCFLAGS)
MFEM_INCFLAGS  ?= -I@MFEM_INC_DIR@ @MFEM_TPLFLAGS@
MFEM_FLAGS     ?= @MFEM_CPPFLAGS@ @MFEM_CXXFLAGS@ @MFEM_INCFLAGS@
MFEM_LIBS      ?= $(ALL_LIBS) $(LDFLAGS)
MFEM_LIB_FILE  ?= @MFEM_LIB_DIR@/libmfem.a
MFEM_BUILD_TAG ?= $(shell uname -snm)
MFEM_PREFIX    ?= $(PREFIX)
MFEM_INC_DIR   ?= $(if $(BUILD_DIR_DEF),@MFEM_BUILD_DIR@,@MFEM_DIR@)
MFEM_LIB_DIR   ?= $(if $(BUILD_DIR_DEF),@MFEM_BUILD_DIR@,@MFEM_DIR@)
# Use "\n" (interpreted by sed) to add a newline.
MFEM_CONFIG_EXTRA ?= $(if $(BUILD_DIR_DEF),MFEM_BUILD_DIR ?= @MFEM_DIR@,)

# If we have 'config' target, export variables used by config/makefile
ifneq (,$(filter config,$(MAKECMDGOALS)))
   export $(MFEM_DEFINES) MFEM_DEFINES $(MFEM_CONFIG_VARS) MFEM_CONFIG_VARS
   export VERBOSE HYPRE_OPT
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
   MFEM_INC_DIR = $(abspath $(PREFIX_INC))
   MFEM_LIB_DIR = $(abspath $(PREFIX_LIB))
   export $(MFEM_DEFINES) MFEM_DEFINES $(MFEM_CONFIG_VARS) MFEM_CONFIG_VARS
   export VERBOSE
endif

# Source dirs in logical order
DIRS = general linalg mesh fem
SOURCE_FILES = $(foreach dir,$(DIRS),$(wildcard $(SRC)$(dir)/*.cpp))
RELSRC_FILES = $(patsubst $(SRC)%,%,$(SOURCE_FILES))
OBJECT_FILES = $(patsubst $(SRC)%,$(BLD)%,$(SOURCE_FILES:.cpp=.o))

.PHONY: lib all clean distclean install config status info deps serial parallel\
 debug pdebug style check test

.SUFFIXES:
.SUFFIXES: .cpp .o
# Remove some default implicit rules
%:	%.o
%.o:	%.cpp
%:	%.cpp

# Default rule.
lib: $(BLD)libmfem.a

# Flags used for compiling all source files.
MFEM_BUILD_FLAGS = $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_TPLFLAGS)\
 $(BUILD_DIR_DEF)

# Rules for compiling all source files.
$(OBJECT_FILES): $(BLD)%.o: $(SRC)%.cpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_BUILD_FLAGS) -c $(<) -o $(@)

all: examples miniapps

.PHONY: miniapps $(EM_DIRS)
miniapps: $(MINIAPP_DIRS)
$(MINIAPP_USE_COMMON): miniapps/common
$(EM_DIRS): lib
	$(MAKE) -C $(BLD)$(@)

.PHONY: doc
doc:
	$(MAKE) -C $(BLD)$(@)

-include $(BLD)deps.mk

$(BLD)libmfem.a: $(OBJECT_FILES)
	$(AR) $(ARFLAGS) $(@) $(OBJECT_FILES)
	$(RANLIB) $(@)

serial debug:    M_MPI=NO
parallel pdebug: M_MPI=YES
serial parallel: M_DBG=NO
debug pdebug:    M_DBG=YES
serial parallel debug pdebug:
	$(MAKE) -f $(THIS_MK) config MFEM_USE_MPI=$(M_MPI) MFEM_DEBUG=$(M_DBG)
	$(MAKE)

deps:
	rm -f $(BLD)deps.mk
	for i in $(RELSRC_FILES:.cpp=); do \
	   $(DEP_CXX) $(MFEM_BUILD_FLAGS) -MM -MT $(BLD)$${i}.o $(SRC)$${i}.cpp\
	      >> $(BLD)deps.mk; done

check: lib
	@printf "Quick-checking the MFEM library."
	@printf " Use 'make test' for more extensive tests.\n"
	@$(MAKE) -C $(BLD)examples \
	$(if $(findstring YES,$(MFEM_USE_MPI)),ex1p-test-par,ex1-test-seq)

test:
	@echo "Testing the MFEM library. This may take a while..."
	@echo "Building all examples and miniapps..."
	@$(MAKE) all
	@ERR=0; for dir in $(EM_TEST_DIRS); do \
	   echo "Running tests in $${dir} ..."; \
	   if ! $(MAKE) -j1 -C $(BLD)$${dir} test; then \
	   ERR=1; fi; done; \
	   if [ 0 -ne $${ERR} ]; then echo "Some tests failed."; exit 1; \
	   else echo "All tests passed."; fi

ALL_CLEAN_SUBDIRS = $(addsuffix /clean,config $(EM_DIRS) doc)
.PHONY: $(ALL_CLEAN_SUBDIRS) miniapps/clean
miniapps/clean: $(addsuffix /clean,$(MINIAPP_DIRS))
$(ALL_CLEAN_SUBDIRS):
	$(MAKE) -C $(BLD)$(@D) $(@F)

clean: $(addsuffix /clean,$(EM_DIRS))
	rm -f $(addprefix $(BLD),*/*.o */*~ *~ libmfem.a deps.mk)

distclean: clean config/clean doc/clean
	rm -rf mfem/

install: $(BLD)libmfem.a
# install static library
	mkdir -p $(PREFIX_LIB)
	$(INSTALL) -m 640 $(BLD)libmfem.a $(PREFIX_LIB)
# install top level includes
	mkdir -p $(PREFIX_INC)
	$(INSTALL) -m 640 $(SRC)mfem.hpp $(SRC)mfem-performance.hpp $(PREFIX_INC)
# install config include
	mkdir -p $(PREFIX_INC)/config
	$(INSTALL) -m 640 $(BLD)config/_config.hpp $(PREFIX_INC)/config/config.hpp
	$(INSTALL) -m 640 $(SRC)config/tconfig.hpp $(PREFIX_INC)/config
# install remaining includes in each subdirectory
	for dir in $(DIRS); do \
	   mkdir -p $(PREFIX_INC)/$$dir && \
	   $(INSTALL) -m 640 $(SRC)$$dir/*.hpp $(PREFIX_INC)/$$dir; done
# install config.mk at root of install tree
	$(MAKE) -C $(BLD)config config-mk CONFIG_MK=config-install.mk
	$(INSTALL) -m 640 $(BLD)config/config-install.mk $(PREFIX)/config.mk
	rm -f $(BLD)config/config-install.mk
# install test.mk at root of install tree
	$(INSTALL) -m 640 $(SRC)config/test.mk $(PREFIX)/test.mk

$(CONFIG_MK):
	$(info )
	$(info MFEM is not configured.)
	$(info Run "make config" first, or see "make help".)
	$(info )
	$(error )

config: $(if $(BUILD_DIR_DEF),build-config,local-config)

.PHONY: local-config
local-config:
	$(MAKE) -C config all
	@printf "\nBuild destination: <source> [$(BUILD_REAL_DIR)]\n\n"

.PHONY: build-config
build-config:
	for d in $(BUILD_SUBDIRS); do mkdir -p $(BLD)$${d}; done
	for dir in "" $(addsuffix /,config $(EM_DIRS) doc); do \
	   printf "# Auto-generated file.\n%s\n%s\n" \
	      "MFEM_DIR = $(MFEM_REAL_DIR)" \
	      "include \$$(MFEM_DIR)/$${dir}makefile" \
	      > $(BLD)$${dir}GNUmakefile; done
	$(MAKE) -C $(BLD)config all
	cd "$(BUILD_DIR)" && ln -sf "$(MFEM_REAL_DIR)/data" .
	for hdr in mfem.hpp mfem-performance.hpp; do \
	   printf "// Auto-generated file.\n%s\n%s\n" \
	   "#define MFEM_BUILD_DIR $(BUILD_REAL_DIR)" \
	   "#include \"$(MFEM_REAL_DIR)/$${hdr}\"" > $(BLD)$${hdr}; done
	@printf "\nBuild destination: $(BUILD_DIR) [$(BUILD_REAL_DIR)]\n\n"

help:
	$(info $(value MFEM_HELP_MSG))
	@true

status info:
	$(info MFEM_VERSION         = $(MFEM_VERSION) [v$(MFEM_VERSION_STRING)])
	$(info MFEM_USE_MPI         = $(MFEM_USE_MPI))
	$(info MFEM_NO_METIS        = $(MFEM_NO_METIS))
	$(info MFEM_USE_METIS_5     = $(MFEM_USE_METIS_5))
	$(info MFEM_DEBUG           = $(MFEM_DEBUG))
	$(info MFEM_USE_GZSTREAM    = $(MFEM_USE_GZSTREAM))
	$(info MFEM_USE_LIBUNWIND   = $(MFEM_USE_LIBUNWIND))
	$(info MFEM_USE_LAPACK      = $(MFEM_USE_LAPACK))
	$(info MFEM_THREAD_SAFE     = $(MFEM_THREAD_SAFE))
	$(info MFEM_USE_OPENMP      = $(MFEM_USE_OPENMP))
	$(info MFEM_USE_MEMALLOC    = $(MFEM_USE_MEMALLOC))
	$(info MFEM_TIMER_TYPE      = $(MFEM_TIMER_TYPE))
	$(info MFEM_USE_SUNDIALS    = $(MFEM_USE_SUNDIALS))
	$(info MFEM_USE_MESQUITE    = $(MFEM_USE_MESQUITE))
	$(info MFEM_USE_SUITESPARSE = $(MFEM_USE_SUITESPARSE))
	$(info MFEM_USE_SUPERLU     = $(MFEM_USE_SUPERLU))
	$(info MFEM_USE_STRUMPACK   = $(MFEM_USE_STRUMPACK))
	$(info MFEM_USE_GECKO       = $(MFEM_USE_GECKO))
	$(info MFEM_USE_GNUTLS      = $(MFEM_USE_GNUTLS))
	$(info MFEM_USE_NETCDF      = $(MFEM_USE_NETCDF))
	$(info MFEM_USE_PETSC       = $(MFEM_USE_PETSC))
	$(info MFEM_USE_MPFR        = $(MFEM_USE_MPFR))
	$(info MFEM_USE_SIDRE       = $(MFEM_USE_SIDRE))
	$(info MFEM_CXX             = $(value MFEM_CXX))
	$(info MFEM_CPPFLAGS        = $(value MFEM_CPPFLAGS))
	$(info MFEM_CXXFLAGS        = $(value MFEM_CXXFLAGS))
	$(info MFEM_TPLFLAGS        = $(value MFEM_TPLFLAGS))
	$(info MFEM_INCFLAGS        = $(value MFEM_INCFLAGS))
	$(info MFEM_FLAGS           = $(value MFEM_FLAGS))
	$(info MFEM_LIBS            = $(value MFEM_LIBS))
	$(info MFEM_LIB_FILE        = $(value MFEM_LIB_FILE))
	$(info MFEM_BUILD_TAG       = $(value MFEM_BUILD_TAG))
	$(info MFEM_PREFIX          = $(value MFEM_PREFIX))
	$(info MFEM_INC_DIR         = $(value MFEM_INC_DIR))
	$(info MFEM_LIB_DIR         = $(value MFEM_LIB_DIR))
	$(info MFEM_BUILD_DIR       = $(MFEM_BUILD_DIR))
	$(info MFEM_MPIEXEC         = $(MFEM_MPIEXEC))
	$(info MFEM_MPIEXEC_NP      = $(MFEM_MPIEXEC_NP))
	@true

ASTYLE = astyle --options=$(SRC)config/mfem.astylerc
FORMAT_FILES = $(foreach dir,$(DIRS) $(EM_DIRS) config,"$(dir)/*.?pp")

style:
	@if ! $(ASTYLE) $(FORMAT_FILES) | grep Formatted; then\
	   echo "No source files were changed.";\
	fi

# Print the contents of a makefile variable, e.g.: 'make print-MFEM_LIBS'.
print-%: ; @printf "%s:\n" $*
	@printf "%s\n" $($*)
