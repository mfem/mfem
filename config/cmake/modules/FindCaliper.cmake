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
#   - CALIPER_FOUND
#   - CALIPER_LIBRARIES
#   - CALIPER_INCLUDE_DIRS

include(MfemCmakeUtilities)
mfem_find_package(Caliper CALIPER CALIPER_DIR
      "include" "caliper/cali.h"
      "lib" "caliper"
      "Paths to headers required by Caliper."
      "Libraries required by Caliper.")

# Append adiak path/lib if the user provided ADIAK_DIR
if(ADIAK_DIR AND EXISTS ${ADIAK_DIR})
    find_package(adiak NO_DEFAULT_PATH REQUIRED PATHS ${ADIAK_DIR}/lib/cmake/adiak ${ADIAK_DIR})
    list(APPEND CALIPER_INCLUDE_DIRS ${adiak_INCLUDE_DIRS})
    list(APPEND CALIPER_LIBRARIES ${adiak_LIBRARIES})
endif()

# Append gotcha path/lib if the user provided GOTCHA_DIR
if(GOTCHA_DIR AND EXISTS ${GOTCHA_DIR})
    find_package(gotcha NO_DEFAULT_PATH REQUIRED PATHS ${GOTCHA_DIR}/lib/cmake/gotcha ${GOTCHA_DIR})
    list(APPEND CALIPER_INCLUDE_DIRS ${gotcha_INCLUDE_DIRS})
    list(APPEND CALIPER_LIBRARIES ${gotcha_LIBRARIES})
endif()

