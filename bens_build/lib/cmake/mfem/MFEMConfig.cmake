# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
# reserved. See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

include(${CMAKE_CURRENT_LIST_DIR}/MFEMConfigVersion.cmake)

set(MFEM_VERSION ${PACKAGE_VERSION})
set(MFEM_VERSION_INT 40001)
set(MFEM_GIT_STRING "heads/AIR_pr-0-g57c5c412a05c1370b7cfc85e34cb73af78a187c5")

set(MFEM_USE_MPI ON)
set(MFEM_USE_METIS ON)
set(MFEM_USE_METIS_5 1)
set(MFEM_DEBUG OFF)
set(MFEM_USE_EXCEPTIONS OFF)
set(MFEM_USE_GZSTREAM OFF)
set(MFEM_USE_LIBUNWIND OFF)
set(MFEM_USE_LAPACK ON)
set(MFEM_THREAD_SAFE OFF)
set(MFEM_USE_OPENMP OFF)
set(MFEM_USE_LEGACY_OPENMP OFF)
set(MFEM_USE_MEMALLOC ON)
set(MFEM_TIMER_TYPE 4)
set(MFEM_USE_SUNDIALS OFF)
set(MFEM_USE_MESQUITE OFF)
set(MFEM_USE_SUITESPARSE OFF)
set(MFEM_USE_SUPERLU OFF)
set(MFEM_USE_STRUMPACK OFF)
set(MFEM_USE_GECKO OFF)
set(MFEM_USE_GNUTLS OFF)
set(MFEM_USE_NETCDF OFF)
set(MFEM_USE_PETSC OFF)
set(MFEM_USE_MPFR OFF)
set(MFEM_USE_SIDRE OFF)
set(MFEM_USE_CONDUIT OFF)
set(MFEM_USE_PUMI OFF)
set(MFEM_USE_CUDA OFF)
set(MFEM_USE_OCCA OFF)
set(MFEM_USE_RAJA OFF)

set(MFEM_CXX_COMPILER "/Library/Developer/CommandLineTools/usr/bin/c++")
set(MFEM_CXX_FLAGS "")


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was MFEMConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(MFEM_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include;/usr/local/Cellar/open-mpi/3.0.0_2/include;/Users/ben/Documents/SoftwareLibraries/metis/include;/Users/ben/Documents/SoftwareLibraries/hypre/src/hypre/include")
foreach (dir ${MFEM_INCLUDE_DIRS})
  set_and_check(MFEM_INCLUDE_DIR "${dir}")
endforeach (dir "${MFEM_INCLUDE_DIRS}")

set_and_check(MFEM_LIBRARY_DIR "${PACKAGE_PREFIX_DIR}/lib")

check_required_components(MFEM)

if (NOT TARGET mfem)
  include(${CMAKE_CURRENT_LIST_DIR}/MFEMTargets.cmake)
endif (NOT TARGET mfem)

set(MFEM_LIBRARIES mfem)
