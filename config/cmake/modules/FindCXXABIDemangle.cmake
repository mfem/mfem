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

# Check for "abi::__cxa_demangle" in <cxxabi.h>.
# Defines the variables:
#    CXXABIDemangle_FOUND
#    CXXABIDemangle_LIBRARIES (if needed)

include(MfemCmakeUtilities)
mfem_find_library(CXXABIDemangle CXXABIDemangle ""
 "Library required for abi::__cxa_demangle." CXXABIDemangle_BUILD
  "
#include <cxxabi.h>
int main()
{
  int demangle_status;
  const char name[] = \"__ZN4mfem10mfem_errorEPKc\";
  char *name_demangle =
    abi::__cxa_demangle(name, NULL, NULL, &demangle_status);
  return 0;
}
")
