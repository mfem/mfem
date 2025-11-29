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
#   - RAJA_FOUND
#   - RAJA_LIBRARIES
#   - RAJA_INCLUDE_DIRS

if (RAJA_FOUND)
   return()
endif()
message(STATUS "Looking for RAJA ...")
if (RAJA_DIR)
   message(STATUS "   in RAJA_DIR = ${RAJA_DIR}")
   find_package(RAJA CONFIG NO_DEFAULT_PATH PATHS "${RAJA_DIR}")
endif()
if (NOT RAJA_FOUND)
   message(STATUS "   in standard CMake locations")
   find_package(RAJA CONFIG)
endif()
if (RAJA_FOUND)
   set(RAJA_LIBRARIES "RAJA" CACHE STRING "RAJA imported target." FORCE)
   set(RAJA_INCLUDE_DIRS "" CACHE STRING "RAJA include dirs (not used)" FORCE)
   message(STATUS
      "Found RAJA target: ${RAJA_LIBRARIES} (version: ${RAJA_VERSION})")
else()
   set(msg STATUS)
   if (RAJA_FIND_REQUIRED)
      set(msg FATAL_ERROR)
   endif()
   message(${msg} "RAJA not found. Please set RAJA_DIR to the RAJA prefix.")
endif()
