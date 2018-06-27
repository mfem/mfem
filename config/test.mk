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

# Utilities for the "make test" and "make check" targets.

# Colors used below:
# green    '\033[0;32m'
# red      '\033[0;31m'
# yellow   '\033[0;33m'
# no color '\033[0m'
COLOR_PRINT = if [ -t 1 ]; then \
   printf $(1)$(2)'\033[0m'$(3); else printf $(2)$(3); fi
PRINT_OK = $(call COLOR_PRINT,'\033[0;32m',OK,"  ($$1 $$2)\n")
PRINT_FAILED = $(call COLOR_PRINT,'\033[0;31m',FAILED,"  ($$1 $$2)\n")
PRINT_SKIP = $(call COLOR_PRINT,'\033[0;33m',SKIP,"\n")

# Timing support
define TIMECMD_detect
timecmd=$$(which time 2> /dev/null);
if [ -n "$$timecmd" ]; then
   if $$timecmd --version > /dev/null 2>&1; then
   echo "$$timecmd" GNU; else echo "$$timecmd" NOTGNU; fi;
else timecmd=$$(command -v time);
   if [ "$$timecmd" = time ]; then
   echo "$$timecmd" BASH; else echo X NONE; fi;
fi
endef
define TIMECMD.GNU
export TIME='%es %MkB %x'; \
set -- $$($(1) $(SHELL) -c "$(2)" 2>&1); while [ "$$#" -gt 3 ]; do shift; done
endef
define TIMECMD.NOTGNU
set -- $$($(1) -l $(SHELL) -c "$(2)" 2>&1; echo $$?); \
set -- "$$1"s "$$(($$7/1024))"kB "$${60}"
endef
define TIMECMD.BASH
TIMEFORMAT=$$'%3Rs'; \
set -- $$({ time $(2); } 2>&1; echo $$?); set -- "$$1" "" "$$2"
endef
define TIMECMD.NONE
$(2); set -- "" "" "$$?"
endef
TIMECMD := $(shell $(TIMECMD_detect))
TIMEFUN := TIMECMD.$(word 2,$(TIMECMD))
TIMECMD := $(word 1,$(TIMECMD))
# Sample use of the timing macro: (returns shell commands as text)
# $(call $(TIMEFUN),$(TIMECMD),$(MY_SHELL_COMMANDS))

ifneq (,$(filter test%,$(MAKECMDGOALS)))
   MAKEFLAGS += -k
endif
# Test runs of the examples/miniapps with parameters - check exit code
mfem-test = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   $(call $(TIMEFUN),$(TIMECMD),$(2) ./$(1) -no-vis $(4) > $(1).stderr 2>&1); \
   if [ "$$3" = 0 ]; \
   then $(PRINT_OK); else $(PRINT_FAILED); cat $(1).stderr; fi; \
   rm -f $(1).stderr; exit $$3

# Test runs of the examples/miniapps - check exit code and if a file exists
mfem-test-file = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   $(call $(TIMEFUN),$(TIMECMD),$(2) ./$(1) -no-vis > $(1).stderr 2>&1); \
   if [ "$$3" = 0 ] && [ -e $(4) ]; \
   then $(PRINT_OK); else $(PRINT_FAILED); cat $(1).stderr; fi; \
   rm -f $(1).stderr; exit $$3

.PHONY: test test-par-YES test-par-NO test-ser test-par test-clean test-print

# What sets of tests to run in serial and parallel
test-par-YES: $(PAR_$(MFEM_TESTS):=-test-par) $(SEQ_$(MFEM_TESTS):=-test-seq)
test-par-NO:  $(SEQ_$(MFEM_TESTS):=-test-seq)
test-ser:     test-par-NO
test-par:     test-par-YES
test:         all test-par-$(MFEM_USE_MPI) clean-exec
test-clean: ; @rm -f *.stderr
test-print: mfem-test=printf "   $(3) [$(2) ./$(1) -no-vis $(if $(4),$(4) )]\n"
test-print: mfem-test-file=printf "   $(3) [$(2) ./$(1) -no-vis ]\n"
test-print: test-par-$(MFEM_USE_MPI)
ifeq ($(MAKECMDGOALS),test-print)
.PHONY: $(PAR_$(MFEM_TESTS)) $(SEQ_$(MFEM_TESTS))
endif

clean-exec: test-clean
