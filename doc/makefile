# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

SHELL = /bin/bash
MFEM_DIR ?= ..
DOXYGEN_CONF = CodeDocumentation.conf


# doxygen uses: graphviz, latex
html: $(DOXYGEN_CONF)
	@# Generate the html documentation
	@doxygen $(DOXYGEN_CONF)
	@echo "<meta http-equiv=\"REFRESH\" content=\"0;URL=CodeDocumentation/html/index.html\">" > CodeDocumentation.html
	@cat warnings.log
	@# Generate the log of undocumented methods
	@( cat $(DOXYGEN_CONF) ; echo "GENERATE_HTML=NO" ; echo "EXTRACT_ALL=NO" ; echo "WARN_LOGFILE=undoc.log" ; echo "QUIET=YES" ) | doxygen - &> /dev/null

clean:
	rm -rf $(DOXYGEN_CONF) CodeDocumentation CodeDocumentation.html *~
	rm -rf  undoc.log warnings.log

$(DOXYGEN_CONF): $(MFEM_DIR)/doc/$(DOXYGEN_CONF).in
	@sed -e 's%@MFEM_SOURCE_DIR@%$(MFEM_DIR)%g' $(<) \
	  > $(DOXYGEN_CONF)

