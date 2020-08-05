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

# Sets the following variables:
#   - SLEPC_FOUND
#   - SLEPC_INCLUDE_DIRS
#   - SLEPC_LIBRARIES

set(SLEPc_REQUIRED_PACKAGES "PETSC" CACHE STRING
  "Additional packages required by SLEPc")

include(MfemCmakeUtilities)
mfem_find_package(SLEPc SLEPC SLEPC_DIR
  "include" "slepceps.h"
  "${PETSC_ARCH}/lib" "slepc" # add NAMES_PER_DIR?
  "Paths to headers required by SLEPc."
  "Libraries required by SLEPc."
  ADD_COMPONENT "config" "${PETSC_ARCH}/include" "slepcconf.h" "" ""
  CHECK_BUILD SLEPC_VERSION_OK TRUE
"
#include \"petsc.h\"
#include \"slepceps.h\"
int main()
{
  PetscErrorCode ierr;
  int argc = 0;
  char** argv = NULL;
  ierr = SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  EPS eps;
  ierr = EPSCreate(PETSC_COMM_SELF, &eps); CHKERRQ(ierr);
  ierr = EPSDestroy(&eps); CHKERRQ(ierr);
  ierr = SlepcFinalize(); CHKERRQ(ierr);
  return 0;
}
"
  )
