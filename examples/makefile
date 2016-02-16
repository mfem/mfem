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
MFEM_DIR = ..
CONFIG_MK = $(MFEM_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_DIR = ../mfem
# CONFIG_MK = $(MFEM_DIR)/config.mk

ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex14
else
   EXAMPLES = ex1p ex2p ex3p ex4p ex5p ex6p ex7p ex8p ex9p ex10p ex11p ex12p ex13p ex14p
endif

.PHONY: all clean

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ ex1 ex1p ex2 ex2p ex3 ex3p ex4 ex4p ex5 ex5p ex6 ex6p ex7 ex7p
	rm -f ex8 ex8p ex9 ex9p ex10 ex10p ex11p ex12p ex13p ex14 ex14p
	rm -f refined.mesh displaced.mesh mesh.* ex5.mesh
	rm -rf Example5* Example9*
	rm -f sphere_refined.* sol.* sol_u.* sol_p.*
	rm -f ex9.mesh ex9-mesh.* ex9-init.* ex9-final.*
	rm -f deformed.* velocity.* elastic_energy.* mode_*
	rm -rf *.dSYM *.TVD.*breakpoints
