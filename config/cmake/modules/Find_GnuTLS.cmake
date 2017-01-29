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

# Find GnuTLS, searching ${GNUTLS_DIR} first. If not successful, tries the
# standard FindGnuTLS.cmake. Defines:
#    GNUTLS_FOUND
#    GNUTLS_INCLUDE_DIRS
#    GNUTLS_LIBRARIES

include(MfemCmakeUtilities)
set(_GnuTLS_REQUIRED_PACKAGES "ALT:" "GnuTLS")
mfem_find_package(_GnuTLS GNUTLS GNUTLS_DIR "include" gnutls/gnutls.h
  "lib" "gnutls;libgnutls"
  "Paths to headers required by GnuTLS." "Libraries required by GnuTLS.")
