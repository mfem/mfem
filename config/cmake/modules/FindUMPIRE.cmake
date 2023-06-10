# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#   - UMPIRE_FOUND
#   - UMPIRE_LIBRARIES
#   - UMPIRE_INCLUDE_DIRS

# It also creates the target (CMake package style) UMPIRE::UMPIRE

if (NOT umpire_DIR AND UMPIRE_DIR)
  set(umpire_DIR ${UMPIRE_DIR}/lib/cmake/umpire)
endif()
message(STATUS "Looking for UMPIRE ...")
message(STATUS "   in UMPIRE_DIR = ${UMPIRE_DIR}")
message(STATUS "      umpire_DIR = ${umpire_DIR}")
find_package(umpire CONFIG)
set(UMPIRE_FOUND ${umpire_FOUND})
set(UMPIRE_LIBRARIES "umpire")
if (UMPIRE_FOUND)
  message(STATUS
    "Found UMPIRE target: ${UMPIRE_LIBRARIES} (version: ${umpire_VERSION})")
else()
  set(msg STATUS)
  if (UMPIRE_FIND_REQUIRED)
    set(msg FATAL_ERROR)
  endif()
  message(${msg}
    "UMPIRE not found. Please set UMPIRE_DIR to the install prefix.")
endif()

if(UMPIRE_FOUND)
  mfem_library_to_package(UMPIRE::UMPIRE ${UMPIRE_INCLUDE_DIRS} ${UMPIRE_LIBRARIES})
endif()
