# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
#   - HDF5_FOUND - If HDF5 was found
#   - HDF5_LIBRARIES - The HDF5 libraries
#   - HDF5_INCLUDE_DIRS - The HDF5 include directories

# First Check for HDF5_DIR
if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "Could not find HDF5. HDF5 support needs explicit HDF5_DIR")
endif()

# Find includes
find_path( HDF5_INCLUDE_DIRS hdf5.h
           PATHS  ${HDF5_DIR}/include/
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( __HDF5_LIBRARY NAMES hdf5 libhdf5 libhdf5_D libhdf5_debug
              PATHS ${HDF5_DIR}/lib
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

find_library( __HDF5_HL_LIBRARY NAMES hdf5_hl libhdf5_hl libhdf5_hl_D libhdf5_hl_debug
              PATHS ${HDF5_DIR}/lib
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

set(HDF5_LIBRARIES ${__HDF5_HL_LIBRARY} ${__HDF5_LIBRARY})

include(FindPackageHandleStandardArgs)

# Handle the QUIETLY and REQUIRED arguments and set HDF5_FOUND to TRUE if all
# listed variables are TRUE
find_package_handle_standard_args(HDF5  DEFAULT_MSG
                                  HDF5_INCLUDE_DIRS
                                  __HDF5_LIBRARY
                                  __HDF5_HL_LIBRARY
                                  HDF5_LIBRARIES )
