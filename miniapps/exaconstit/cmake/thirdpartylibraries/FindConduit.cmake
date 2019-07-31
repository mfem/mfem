###############################################################################
#
# Setup Conduit
# This file defines:
#  CONDUIT_FOUND is set if it's been found
#  CONDUIT_INCLUDE_DIRS - The Conduit include directories
#  CONDUIT_LIBRARY - The Conduit library

# first check for CONDUIT_DIR

if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit support needs explicit CONDUIT_DIR")
endif()

find_package(Conduit REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CONDUIT_DIR}/lib/cmake)

if(CONDUIT_RELAY_HDF5_ENABLED)
   set(HDF5_ROOT ${CONDUIT_HDF5_DIR})
   find_package(HDF5 REQUIRED)
   message("HDF5 Libraries found: ${HDF5_LIBRARIES}")
else()
   set(HDF5_LIBRARIES "")
endif()