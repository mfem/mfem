###############################################################################
#
# Setup MFEM
# This file defines:
#  MFEM_FOUND - If MFEM was found
#  MFEM_INCLUDE_DIRS - The MFEM include directories
#  MFEM_LIBRARY - The MFEM library

# Since this is a miniapp of mfem this probably should be as complicated
# but it is.
# first Check for MFEM_DIR

if(NOT MFEM_DIR)
    MESSAGE(FATAL_ERROR "Could not find MFEM. MFEM support needs explicit MFEM_DIR")
endif()

find_package(MFEM)

#This is incredibly hacky and probably should be replaced with something more robust down the road.
#However, MFEM already supplies us with all the flags we need to compile our stuff, so we could just
#use this to simplify having to go hunt down where hypre and metis are located...
file(STRINGS ${MFEM_DIR}/../../../share/mfem/config.mk ConfigContents)
foreach(NameAndValue ${ConfigContents})
  # Strip leading spaces
  #string(STRIP ${NameAndValue} NameAndValue)
  string(REGEX REPLACE "^[ ]+" "" NameAndValue ${NameAndValue})
  string(STRIP ${NameAndValue} NameAndValue)
  # Find variable name
  string(REGEX MATCH "^[^=]+" Name ${NameAndValue})
  # Find the value
  # Strip following spaces
  string(REPLACE "${Name}=" "" Value ${NameAndValue})
  string(STRIP ${Name} Name)
  # Set the variable
  string(COMPARE EQUAL "${Name}" "MFEM_EXT_LIBS" results_cmp)
    
  if(results_cmp)
     string(STRIP ${Value} Value)
     set(${Name} "${Value}")
  endif()
endforeach()

#Print out what the various flags are that we might need
#message(" MFEM external library flags ... ${MFEM_EXT_LIBS}")

#Eventually we'll need to set this up so that various flags are turned on based
#on what MFEM was compiled with.

#find includes
 find_path( MFEM_INCLUDE_DIRS mfem.hpp
            PATHS  ${MFEM_DIR}/../../../include/ ${MFEM_DIR}
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)

find_library( MFEM_LIBRARY NAMES mfem libmfem
              PATHS ${MFEM_LIBRARY_DIR}
              NO_DEFAULT_PATH
              NO_CMAKE_ENVIRONMENT_PATH
              NO_CMAKE_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set RAJA_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MFEM  DEFAULT_MSG
                                  MFEM_INCLUDE_DIRS
                                  MFEM_LIBRARY )
 