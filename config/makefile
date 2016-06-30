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

# Helper print-info function
mfem-info = $(if $(filter YES,$(VERBOSE)),$(info *** [info]$(1)),)

# The list of all defines that may be enabled in config.hpp and config.mk
# is kept in the MFEM_DEFINES variable, exported here from ../makefile

# The list of variables to set in config.mk using their current values
# is kept in the MFEM_CONFIG_VARS variable, exported here from ../makefile

# Debug output
# $(foreach def,$(MFEM_DEFINES),$(info $(def) = $(value $(def))))
# $(foreach var,$(MFEM_CONFIG_VARS),$(info $(var) = $(value $(var))))

# Output path + file name for the config header
CONFIG_HPP = config.hpp

# Output path + file name for the config makefile
CONFIG_MK = config.mk

.SUFFIXES:
.PHONY: all header config-mk

all: header config-mk

header:
	$(call mfem-info, Writing $(CONFIG_HPP) ...)
	@cp -f config.hpp.in $(CONFIG_HPP) && \
	set -- && \
	for def in $${MFEM_DEFINES}; do \
	   eval var=\$$$$def && \
	   if [ "NO" != "$${var}" ]; then \
	      set -- "$$@" "-e" "s|// \(#define $${def}\)|\1|" && \
	      set -- "$$@" "-e" "s#@$${def}@#$${var}#g"; \
           fi; \
	done && \
	sed "$$@" $(CONFIG_HPP) > $(CONFIG_HPP).tmp && \
	mv -f $(CONFIG_HPP).tmp $(CONFIG_HPP)

# The target below uses the values of the shell variables listed in
# the shell variables MFEM_DEFINES and MFEM_CONFIG_VARS
config-mk:
	$(call mfem-info, Writing $(CONFIG_MK) ...)
	@cp -f config.mk.in $(CONFIG_MK) && \
	set -- && \
	for ref in $${MFEM_DEFINES} $${MFEM_CONFIG_VARS}; do \
	   eval var=\$$$$ref && \
	   set -- "$$@" "-e" "s#@$${ref}@#$${var}#g"; \
	done && \
	set -- "$$@" "-e" 's/@\([^@]*\)@/$$(\1)/g' "-e" 's/ *$$//g' && \
	sed "$$@" $(CONFIG_MK) > $(CONFIG_MK).tmp && \
	mv -f $(CONFIG_MK).tmp $(CONFIG_MK)

clean:
	rm -f $(CONFIG_HPP) $(CONFIG_MK)
