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
