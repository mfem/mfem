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

MFEM_DIR ?= ..
DOXYGEN_CONG = CodeDocumentation.conf

# doxygen uses: graphviz, latex
html: $(DOXYGEN_CONG)
	doxygen $(DOXYGEN_CONG)
	rm -f CodeDocumentation.html
	ln -s CodeDocumentation/html/index.html CodeDocumentation.html

clean:
	rm -rf $(DOXYGEN_CONG) CodeDocumentation CodeDocumentation.html *~

$(DOXYGEN_CONG): $(MFEM_DIR)/doc/$(DOXYGEN_CONG).in
	sed -e 's%@MFEM_SOURCE_DIR@%$(MFEM_DIR)%g' $(<) \
	  > $(DOXYGEN_CONG)
