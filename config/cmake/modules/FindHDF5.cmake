###############################################################################
#
# This is a copy of the FincHDF5.cmake file from ALE3D. This is need to build 
# on macOS. If MFEM provides a FindHDF5.cmake in the future, use that instead. 
#
# Setup HDF5
# This file defines:
#  HDF5_FOUND - If HDF5 was found
#  HDF5_INCLUDE_DIRS - The HDF5 include directories
#  HDF5_LIBRARIES - The HDF5 libraries

# first Check for HDF5_DIR

if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "Could not find HDF5. HDF5 support needs explicit HDF5_DIR")
endif()

#find includes
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
# handle the QUIETLY and REQUIRED arguments and set HDF5_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(HDF5  DEFAULT_MSG
                                  HDF5_INCLUDE_DIRS
                                  __HDF5_LIBRARY
                                  __HDF5_HL_LIBRARY
                                  HDF5_LIBRARIES )
