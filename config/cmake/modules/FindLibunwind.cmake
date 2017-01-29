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

# Find libunwind.
# Defines the following variables:
#   - LIBUNWIND_FOUND
#   - LIBUNWIND_LIBRARIES    (if needed)
#   - LIBUNWIND_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
set(Libunwind_SKIP_FPHSA TRUE)
mfem_find_library(Libunwind LIBUNWIND "unwind" "The libunwind library."
  LIBUNWIND_BUILD
  "
#define UNW_LOCAL_ONLY
#include <libunwind.h>
int main()
{
  unw_context_t uc;
  unw_getcontext(&uc);
  return 0;
}
")
unset(Libunwind_SKIP_FPHSA)

set(Libunwind_SKIP_LOOKING_MSG TRUE)
mfem_find_package(Libunwind LIBUNWIND LIBUNWIND_DIR "include" libunwind.h
  "lib" unwind
  "Paths to headers required by libunwind." "Libraries required by libunwind.")
unset(Libunwind_SKIP_LOOKING_MSG)
