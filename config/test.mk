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
# no color '\033[0m'
COLOR_PRINT = if [ -t 1 ]; then \
   printf $(1)$(2)'\033[0m'$(3); else printf $(2)$(3); fi
PRINT_OK = $(call COLOR_PRINT,'\033[0;32m',OK,"\n")
PRINT_FAILED = $(call COLOR_PRINT,'\033[0;31m',FAILED,"\n")

# Test runs of the examples/miniapps with parameters - check exit code
mfem-test = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   if ($(2) ./$(1) -no-vis $(4) > /dev/null); \
   then $(PRINT_OK); else $(PRINT_FAILED); fi

# Test runs of the examples/miniapps - check exit code and if a file exists
mfem-test-file = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   if ($(2) ./$(1) -no-vis > /dev/null) && [ -e $(4) ]; \
   then $(PRINT_OK); else $(PRINT_FAILED); fi

.PHONY: test test-par-YES test-par-NO

# What sets of tests to run in serial and parallel
test-par-YES: $(PAR_$(MFEM_TESTS):=-test-par) $(SEQ_$(MFEM_TESTS):=-test-seq)
test-par-NO:  $(SEQ_$(MFEM_TESTS):=-test-seq)
test:         all test-par-$(MFEM_USE_MPI) clean-exec
