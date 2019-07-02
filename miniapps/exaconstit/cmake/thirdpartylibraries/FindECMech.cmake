###############################################################################
#
# Setup ExaCMech
# This file defines:
#  ECMECH_FOUND - If ExaCMech was found
#  ECMECH_INCLUDE_DIRS - The ExaCMech include directories
#  ECMECH_LIBRARY - The ExaCMech library

# first check for ExaCMech_DIR

if(NOT ECMECH_DIR)
    MESSAGE(FATAL_ERROR "Could not find ExaCMech. ExaCMech support needs explicit ECMECH_DIR")
endif()

find_path( ECMECH_INCLUDE_DIRS ECMech_core.h
           PATHS  ${ECMECH_DIR}/include/ ${ECMECH_DIR}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( ECMECH_LIBRARY NAMES ecmech libecmech
              PATHS ${ECMECH_DIR}/lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ECMECH  DEFAULT_MSG
                                  ECMECH_INCLUDE_DIRS
                                  ECMECH_LIBRARY)
