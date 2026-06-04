# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. All Rights reserved.
# See files LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

.PHONY: all clean

all: tied-poisson

tied-poisson: tied-poisson.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) tied-poisson.cpp solver_utils.cpp -o tied-poisson $(MFEM_LIBS) -framework Accelerate

clean:
	rm -f tied-poisson *.o *~ tied-poisson-mesh*.mesh tied-poisson-sol*.gf
	rm -rf *.dSYM *.TVD.*breakpoints
