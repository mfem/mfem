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
