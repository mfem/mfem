# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

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
set -- $$($(1) -l $(SHELL) -c "{ $(2); } > /dev/null 2>&1" 2>&1; echo $$?); \
set -- "$$1"s "$$(($$7/1024))"kB "$${!#}"
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
# Test runs of the examples/miniapps with parameters - check exit code:
# 0 means success, 242 means the test was skipped, anything else means error
mfem-test = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   $(call $(TIMEFUN),$(TIMECMD),$(2) ./$(1) $(if $(5),,-no-vis )$(4) \
     > $(1).stderr 2>&1); \
   err="$$3"; \
   if [ "$$3" = 0 ]; then $(PRINT_OK); \
   else if [ "$$3" = 242 ]; then $(PRINT_SKIP); err=0; \
   else $(PRINT_FAILED); cat $(1).stderr; fi; fi; \
   rm -f $(1).stderr; exit $$err

# Test runs of the examples/miniapps - check exit code and if a file exists
# See mfem-test for the interpretation of the error code
mfem-test-file = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   $(call $(TIMEFUN),$(TIMECMD),$(2) ./$(1) -no-vis > $(1).stderr 2>&1); \
   err="$$3"; \
   if [ "$$3" = 0 ] && [ -e $(4) ]; then $(PRINT_OK); \
   else if [ "$$3" = 242 ] && [ -e $(4) ]; then $(PRINT_SKIP); err=0; \
   else $(PRINT_FAILED); cat $(1).stderr; err=64; fi; fi; \
   rm -f $(1).stderr; exit $$err

.PHONY: test test-par-YES test-par-NO test-ser test-par test-clean test-print

# What sets of tests to run in serial and parallel
test-par-YES: $(PAR_$(MFEM_TESTS):=-test-par) $(SEQ_$(MFEM_TESTS):=-test-seq)
test-par-NO:  $(SEQ_$(MFEM_TESTS):=-test-seq)
ifeq ($(MFEM_USE_CUDA),YES)
.PHONY: test-par-YES-cuda test-par-NO-cuda test-ser-cuda test-par-cuda test-cuda
test-par-YES: test-par-YES-cuda
test-par-NO:  test-par-NO-cuda
test-par-YES-cuda: test-par-cuda test-ser-cuda
test-par-NO-cuda:  test-ser-cuda
test-ser-cuda: $(SEQ_DEVICE_$(MFEM_TESTS):=-test-seq-cuda)
test-par-cuda: $(PAR_DEVICE_$(MFEM_TESTS):=-test-par-cuda)
test-cuda: test-par-$(MFEM_USE_MPI)-cuda clean-exec
endif
ifeq ($(MFEM_USE_HIP),YES)
.PHONY: test-par-YES-hip test-par-NO-hip test-ser-hip test-par-hip test-hip
test-par-YES: test-par-YES-hip
test-par-NO:  test-par-NO-hip
test-par-YES-hip: test-par-hip test-ser-hip
test-par-NO-hip:  test-ser-hip
test-ser-hip: $(SEQ_DEVICE_$(MFEM_TESTS):=-test-seq-hip)
test-par-hip: $(PAR_DEVICE_$(MFEM_TESTS):=-test-par-hip)
test-hip: test-par-$(MFEM_USE_MPI)-hip clean-exec
endif
test-ser:     test-par-NO
test-par:     test-par-YES
test:         all test-par-$(MFEM_USE_MPI) clean-exec
test-noclean: all test-par-$(MFEM_USE_MPI)
test-clean: ; @rm -f *.stderr
test-print: \
 mfem-test=printf "   $(3) [$(2) ./$(1) $(if $(5),,-no-vis )$(if $(4),$(4) )]\n"
test-print: mfem-test-file=printf "   $(3) [$(2) ./$(1) -no-vis ]\n"
test-print: test-par-$(MFEM_USE_MPI)
ifeq ($(MAKECMDGOALS),test-print)
.PHONY: $(PAR_$(MFEM_TESTS)) $(SEQ_$(MFEM_TESTS))
endif

clean-exec: test-clean
