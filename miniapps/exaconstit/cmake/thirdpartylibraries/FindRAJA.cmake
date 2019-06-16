###############################################################################
#
# Setup RAJA
# This file defines:
#  RAJA_FOUND - If RAJA was found
#  RAJA_INCLUDE_DIRS - The RAJA include directories
#  RAJA_LIBRARY - The RAJA library

# first Check for RAJA_DIR

if(NOT RAJA_DIR)
    MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA support needs explicit RAJA_DIR")
endif()

find_package(RAJA)

if(RAJA_CONFIG_LOADED)

   if(ENABLE_OPENMP)
      set(BLT_CXX_FLAGS "${BLT_CXX_FLAGS} -fopenmp" CACHE PATH "")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" CACHE STRING "" FORCE)
   endif()

endif()

set(RAJA_INCLUDE_DIRS ${RAJA_INCLUDE_DIR})

#find includes
#find_path( RAJA_INCLUDE_DIRS RAJA/RAJA.hpp
#           PATHS  ${RAJA_DIR}/include/ ${RAJA_DIR}
#           NO_DEFAULT_PATH
#           NO_CMAKE_ENVIRONMENT_PATH
#           NO_CMAKE_PATH
#           NO_SYSTEM_ENVIRONMENT_PATH
#           NO_CMAKE_SYSTEM_PATH)

find_library( RAJA_LIBRARY NAMES RAJA libRAJA
              PATHS ${RAJA_LIB_DIR}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RAJA  DEFAULT_MSG
                                  RAJA_INCLUDE_DIRS
                                  RAJA_LIBRARY )
