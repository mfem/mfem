###############################################################################
#
# Setup ExaCMech
# This file defines:
#  HYPRE_FOUND - If HYPRE was found
#  HYPRE_INCLUDE_DIRS - The HYPRE include directories
#  HYPRE_LIBRARY - The HYPRE library

# first check for ExaCMech_DIR

if(NOT HYPRE_DIR)
    MESSAGE(FATAL_ERROR "Could not find HYPRE. HYPRE support needs explicit HYPRE_DIR")
endif()

find_path( HYPRE_INCLUDE_DIRS HYPRE.h
           PATHS  ${HYPRE_DIR}/include/ ${HYPRE_DIR}
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

find_library( HYPRE_LIBRARY NAMES HYPRE libHYPRE
              PATHS ${HYPRE_DIR}/lib/
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(HYPRE  DEFAULT_MSG
                                  HYPRE_INCLUDE_DIRS
                                  HYPRE_LIBRARY)
