# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables if fetching of TPLs is disabled (default):
#   - METIS_FOUND
#   - METIS_LIBRARIES
#   - METIS_INCLUDE_DIRS
#   - METIS_VERSION_5
# otherwise, the following are defined:
#   - METIS (imported library target)
#   - METIS_VERSION_5 (cache variable)

if (METIS_FETCH OR FETCH_TPLS)
  set(METIS_FETCH_VERSION 4.0.3)
  add_library(METIS STATIC IMPORTED)
  # define external project
  message(STATUS "Will fetch METIS ${METIS_FETCH_VERSION} to be built with default options")
  set(PREFIX ${CMAKE_BINARY_DIR}/fetch/metis)
  include(ExternalProject)
  ExternalProject_Add(metis
    GIT_REPOSITORY https://github.com/mfem/tpls
    GIT_TAG b60352fbe9675d374b00828055e55be4584c7995 # tag from 1/16/25
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
    PREFIX ${PREFIX}
    CONFIGURE_COMMAND tar -xzf ../metis/metis-${METIS_FETCH_VERSION}-mac.tgz --strip=1
    BUILD_COMMAND $(MAKE) COPTIONS=-Wno-incompatible-pointer-types
    INSTALL_COMMAND mkdir -p ${PREFIX}/lib && cp libmetis.a ${PREFIX}/lib/)
  # set imported library target properties
  add_dependencies(METIS metis)
  set_target_properties(METIS PROPERTIES
    IMPORTED_LOCATION ${PREFIX}/lib/libmetis.a)
  # set cache variables that would otherwise be set after mfem_find_package call
  set(METIS_VERSION_5 FALSE CACHE BOOL "Is METIS version 5?")
  return()
endif()

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
    idx_t n = 10;
    idx_t nparts = 5;
    idx_t edgecut;
    idx_t* partitioning = new idx_t[10];
    idx_t* I = partitioning,
       * J = partitioning;

    idx_t ncon = 1;
    int err;
    idx_t options[40];

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
