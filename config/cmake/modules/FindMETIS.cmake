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

# Defines the following variables:
#   - METIS_FOUND
#   - METIS_LIBRARIES
#   - METIS_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(METIS METIS METIS_DIR "include;Lib" "metis.h"
  "lib" "metis;metis4;metis5"
  "Paths to headers required by METIS." "Libraries required by METIS."
  CHECK_BUILD METIS_VERSION_5 FALSE
  "
#include <metis.h>
#include <cstddef> // So NULL is defined

int main()
{
    int n = 10;
    int nparts = 5;
    int edgecut;
    int* partitioning = new int[10];
    int* I = partitioning,
       * J = partitioning;

    int ncon = 1;
    int err;
    int options[40];

    METIS_SetDefaultOptions(options);
    options[10] = 1; // set METIS_OPTION_CONTIG

    err = METIS_PartGraphKway(&n,
                              &ncon,
                              I,
                              J,
                              (idx_t *) NULL,
                              (idx_t *) NULL,
                              (idx_t *) NULL,
                              &nparts,
                              (real_t *) NULL,
                              (real_t *) NULL,
                              options,
                              &edgecut,
                              partitioning);
    return err;
}
")

# Expose METIS_VERSION_5 (it is created as INTERNAL) and copy its value to
# MFEM_USE_METIS_5:
set(MFEM_USE_METIS_5 ${METIS_VERSION_5})
unset(METIS_VERSION_5 CACHE)
set(METIS_VERSION_5 ${MFEM_USE_METIS_5} CACHE BOOL "Is METIS version 5?")
