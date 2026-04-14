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

# Defines the following variables:
#   - CAMP_FOUND
#   - CAMP_LIBRARIES
#   - CAMP_INCLUDE_DIRS

include(MfemCmakeUtilities)

mfem_find_package(CAMP CAMP CAMP_DIR
      "include" "camp/camp.hpp"
      "lib" "camp"
      "Paths to headers required by CAMP."
      "Libraries required by CAMP.")

# RAJA commonly lists "camp" in INTERFACE_LINK_LIBRARIES. If there is no CMake
# target named "camp", CMake treats it as a bare library name (-lcamp).
if (CAMP_FOUND AND NOT TARGET camp)
   list(GET CAMP_LIBRARIES 0 _camp_lib0)
   add_library(camp UNKNOWN IMPORTED)
   set_target_properties(camp PROPERTIES
      IMPORTED_LOCATION "${_camp_lib0}"
      INTERFACE_INCLUDE_DIRECTORIES "${CAMP_INCLUDE_DIRS}")
   set(CAMP_LIBRARIES "camp" CACHE STRING "CAMP imported target." FORCE)
   unset(_camp_lib0)
endif()

