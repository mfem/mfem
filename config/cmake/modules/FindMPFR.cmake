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

# Find MPFR.
# Defines the following variables:
#   - MPFR_FOUND
#   - MPFR_LIBRARIES    (if needed)
#   - MPFR_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
set(MPFR_SKIP_STANDARD TRUE)
set(MPFR_SKIP_FPHSA TRUE)
mfem_find_library(MPFR MPFR "mpfr" "The MPFR library." MPFR_BUILD
  "
#include <mpfr.h>
int main()
{
  mpfr_t one;
  mpfr_init2(one, 128);
  mpfr_set_si(one, 1, GMP_RNDN);
  mpfr_clear(one);
  return 0;
}
")
unset(MPFR_SKIP_FPHSA)
unset(MPFR_SKIP_STANDARD)

set(MPFR_SKIP_LOOKING_MSG TRUE)
mfem_find_package(MPFR MPFR MPFR_DIR "include" mpfr.h "lib" mpfr
  "Paths to headers required by MPFR." "Libraries required by MPFR.")
unset(MPFR_SKIP_LOOKING_MSG)
