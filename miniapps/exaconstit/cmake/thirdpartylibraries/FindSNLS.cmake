###############################################################################
#
# Setup SNLS
# This file defines:
#  SNLS_FOUND - If SNLS was found
#  SNLS_INCLUDE_DIRS - The SNLS include directories
#  SNLS_LIBRARY - The SNLS library

# first check for SNLS_DIR

if(NOT SNLS_DIR)
    MESSAGE(FATAL_ERROR "Could not find SNLS. SNLS support needs explicit SNLS_DIR")
endif()

find_path( SNLS_INCLUDE_DIRS SNLS_port.h
           PATHS  ${SNLS_DIR}/include/ ${SNLS_DIR}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( SNLS_LIBRARY NAMES snls libsnls
              PATHS ${SNLS_DIR}/lib/ ${ECMECH_DIR}/lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(SNLS  DEFAULT_MSG
                                  SNLS_INCLUDE_DIRS
                                  SNLS_LIBRARY)
