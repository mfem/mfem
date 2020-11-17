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

# Find libdl.
# Defines:
#    LIBDL_FOUND
#    LIBDL_LIBRARIES (if needed)

include(MfemCmakeUtilities)
mfem_find_library(LIBDL LIBDL "dl" "The dynamic library." LIBDL_BUILD
  "
#define _GNU_SOURCE
#include <dlfcn.h>
int main()
{
  Dl_info info;
  dladdr((void*)0, &info);
  return 0;
}
")
