#------------------------------------------------------------------------------#
# Distributed under the OSI-approved Apache License, Version 2.0.  See
# accompanying file Copyright.txt for details.
#------------------------------------------------------------------------------#
#
# FindADIOS2
# -----------
#
# Try to find the ADIOS2 library
#
# This module defines the following variables:
#
#   ADIOS2_FOUND        - System has ADIOS2
#   ADIOS2_INCLUDE_DIRS - The ADIOS2 include directory
#   ADIOS2_LIBRARIES    - Link these to use ADIOS2
#
# and the following imported targets:
#   ADIOS2::ADIOS2 - The ADIOS2 compression library target
#
# You can also set the following variable to help guide the search:
#   ADIOS2_DIR - The install prefix for ADIOS2 containing the
#              include and lib folders
#              Note: this can be set as a CMake variable or an
#                    environment variable.  If specified as a CMake
#                    variable, it will override any setting specified
#                    as an environment variable.

if(NOT ADIOS2_FOUND)
  if((NOT ADIOS2_DIR) AND (NOT (ENV{ADIOS2_DIR} STREQUAL "")))
    set(ADIOS2_DIR "$ENV{ADIOS2_DIR}")
  endif()
  if(ADIOS2_DIR)
    set(ADIOS2_INCLUDE_OPTS HINTS ${ADIOS2_DIR}/include NO_DEFAULT_PATHS)
    set(ADIOS2_LIBRARY_OPTS
      HINTS ${ADIOS2_DIR}/lib ${ADIOS2_DIR}/lib64
      NO_DEFAULT_PATHS
    )
  endif()

  find_path(ADIOS2_INCLUDE_DIR adios2.h ${ADIOS2_INCLUDE_OPTS})
  
  # adios2 version 2.5.0 
  find_library(ADIOS2_LIBRARY NAMES adios2 ${ADIOS2_LIBRARY_OPTS})
  
  # adios2 version 2.6.0 and onwards
  if(NOT ADIOS2_LIBRARY)
  	find_library(ADIOS2_CXX11_MPI_LIBRARY NAMES adios2_cxx11_mpi ${ADIOS2_LIBRARY_OPTS})
  	find_library(ADIOS2_CXX11_LIBRARY NAMES adios2_cxx11 ${ADIOS2_LIBRARY_OPTS})
  	set(ADIOS2_LIBRARY ${ADIOS2_CXX11_MPI_LIBRARY} ${ADIOS2_CXX11_LIBRARY})
  	if(MFEM_USE_MPI)
  	  add_definitions(-DADIOS2_USE_MPI)
  	endif()
  endif() 

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ADIOS2
    FOUND_VAR ADIOS2_FOUND
    REQUIRED_VARS ADIOS2_LIBRARY ADIOS2_INCLUDE_DIR
  )
  if(ADIOS2_FOUND)
    set(ADIOS2_INCLUDE_DIRS ${ADIOS2_INCLUDE_DIR})
    set(ADIOS2_LIBRARIES ${ADIOS2_LIBRARY})
  endif()
endif()
