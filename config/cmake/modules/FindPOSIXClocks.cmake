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

# Find POSIX clocks.
# Defines:
#    POSIXCLOCKS_FOUND
#    POSIXCLOCKS_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_library(POSIXClocks POSIXCLOCKS "rt"
  "Library required by POSIX clocks." POSIXCLOCKS_BUILD
  "
#include <time.h>
int main()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 0;
}
")
