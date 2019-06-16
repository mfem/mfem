###############################################################################
#
# Setup METIS
# This file defines:
#  METIS_FOUND - If METIS was found
#  METIS_INCLUDE_DIRS - The METIS include directories
#  METIS_LIBRARY - The ExaCMech library

# first check for METIS_DIR

if(NOT METIS_DIR)
    MESSAGE(FATAL_ERROR "Could not find METIS. METIS support needs explicit METIS_DIR")
endif()

find_path( METIS_INCLUDE_DIRS metis.h
           PATHS  ${METIS_DIR}/Lib/ ${METIS_DIR}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( METIS_LIBRARY NAMES metis libmetis
              PATHS ${METIS_DIR}/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(METIS  DEFAULT_MSG
                                  METIS_INCLUDE_DIRS
                                  METIS_LIBRARY)
