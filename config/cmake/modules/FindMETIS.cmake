# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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

if (FETCH_TPLS)
  message(STATUS "Fetching metis from GitHub ...")
  include(FetchContent)
  FetchContent_Declare(METIS
    GIT_REPOSITORY https://github.com/mfem/tpls
    GIT_TAG b60352fbe9675d374b00828055e55be4584c7995 # tag from 1/16/25
    GIT_SHALLOW TRUE
    BINARY_DIR fetch/metis
    OVERRIDE_FIND_PACKAGE
  )
  # obtain source and create build directory, namely so include directory is
  # present to pass MFEM checks at end of configuation step
  FetchContent_MakeAvailable(METIS)
  execute_process(COMMAND mkdir ${metis_BINARY_DIR}/include)
  execute_process(COMMAND mkdir ${metis_BINARY_DIR}/Lib)
  # create custom command and target for building
  add_custom_command(OUTPUT ${metis_BINARY_DIR}/Lib/libmetis.a
    COMMAND cd ${metis_SOURCE_DIR} && tar -xzf metis-4.0.3.tar.gz
    COMMAND cd ${metis_SOURCE_DIR}/metis-4.0.3 && make -j
    COMMAND cp ${metis_SOURCE_DIR}/metis-4.0.3/Lib/*.h ${metis_BINARY_DIR}/include/
    COMMAND cp ${metis_SOURCE_DIR}/metis-4.0.3/libmetis.a ${metis_BINARY_DIR}/Lib/
    COMMENT "Building metis ..."
  )
  add_custom_target(METIS_LIBRARY DEPENDS ${metis_BINARY_DIR}/Lib/libmetis.a)
  # create interface library target for linking
  add_library(METIS INTERFACE IMPORTED)
  add_dependencies(METIS METIS_LIBRARY)
  target_link_libraries(METIS INTERFACE ${metis_BINARY_DIR}/Lib/libmetis.a)
  target_include_directories(METIS INTERFACE ${metis_BINARY_DIR}/include)
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
