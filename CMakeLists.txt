# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

cmake_minimum_required(VERSION 2.8.11)
set(USER_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/config/user.cmake" CACHE PATH
  "Path to optional user configuration file.")

# Load user settings before the defaults - this way the defaults will not
# overwrite the user set options. If the user has not set all options, we still
# have the defaults.
message(STATUS "(optional) USER_CONFIG = ${USER_CONFIG}")
include("${USER_CONFIG}" OPTIONAL)
include("${CMAKE_CURRENT_SOURCE_DIR}/config/defaults.cmake")

# Allow overwriting of the compiler by setting CXX/MPICXX on the command line or
# in user.cmake.
if (NOT CMAKE_CXX_COMPILER)
  if (CXX)
    set(CMAKE_CXX_COMPILER ${CXX})
    # Avoid some issues when CXX is defined
    unset(CXX)
    unset(CXX CACHE)
  endif()
  if (MFEM_USE_MPI AND MPICXX)
    # In parallel MPICXX takes precedence, if defined.
    set(CMAKE_CXX_COMPILER ${MPICXX})
    # Setting the variables below circumvents autodetection, see FindMPI.cmake.
    set(MPI_CXX_INCLUDE_PATH "")
    set(MPI_CXX_LIBRARIES "")
  endif()
endif()

#-------------------------------------------------------------------------------
# Project name and version
#-------------------------------------------------------------------------------
project(mfem NONE)
# Current version of MFEM, see also `makefile`.
#   mfem_VERSION = (string)
#   MFEM_VERSION = (int)   [automatically derived from mfem_VERSION]
set(${PROJECT_NAME}_VERSION 3.4.1)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(FATAL_ERROR
    "MFEM does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

# Set xSDK defaults.
set(USE_XSDK_DEFAULTS_DEFAULT OFF)
set(XSDK_ENABLE_CXX ON)
set(XSDK_ENABLE_C OFF)
set(XSDK_ENABLE_Fortran OFF)

# Check if we need to enable C or Fortran.
if (CMAKE_VERSION VERSION_LESS 3.2 OR
    MFEM_USE_CONDUIT OR
    MFEM_USE_SIDRE OR
    MFEM_USE_PETSC)
   # This seems to be needed by:
   #  * find_package(BLAS REQUIRED) and
   #  * find_package(HDF5 REQUIRED) needed, in turn, by:
   #    - find_package(AXOM REQUIRED)
   #  * find_package(PETSc REQUIRED)
   set(XSDK_ENABLE_C ON)
endif()
if (MFEM_USE_STRUMPACK)
  # Just needed to find the MPI_Fortran libraries to link with
  set(XSDK_ENABLE_Fortran ON)
endif()

# Include xSDK default CMake file.
include("${CMAKE_CURRENT_SOURCE_DIR}/config/XSDKDefaults.cmake")

# Enable languages.
enable_language(CXX)
if (XSDK_ENABLE_C)
   enable_language(C)
endif()
if (XSDK_ENABLE_Fortran)
   enable_language(Fortran)
endif()

# Suppress warnings about MACOSX_RPATH
set(CMAKE_MACOSX_RPATH OFF CACHE BOOL "")

# CMake needs to know where to find things
set(MFEM_CMAKE_PATH ${PROJECT_SOURCE_DIR}/config)
set(CMAKE_MODULE_PATH ${MFEM_CMAKE_PATH}/cmake/modules)

# Load MFEM CMake utilities.
include(MfemCmakeUtilities)

string(TOUPPER "${PROJECT_NAME}" PROJECT_NAME_UC)
mfem_version_to_int(${${PROJECT_NAME}_VERSION} ${PROJECT_NAME_UC}_VERSION)
set(${PROJECT_NAME_UC}_VERSION_STRING ${${PROJECT_NAME}_VERSION})
if (EXISTS ${PROJECT_SOURCE_DIR}/.git)
   execute_process(
     COMMAND git describe --all --long --abbrev=40 --dirty --always
     WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
     OUTPUT_VARIABLE ${PROJECT_NAME_UC}_GIT_STRING
     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if (NOT ${PROJECT_NAME_UC}_GIT_STRING)
   set(${PROJECT_NAME_UC}_GIT_STRING "(unknown)")
endif()

#-------------------------------------------------------------------------------
# Process configuration options
#-------------------------------------------------------------------------------

# MFEM_DEBUG
if (CMAKE_BUILD_TYPE MATCHES "Debug|debug|DEBUG")
  set(MFEM_DEBUG ON)
else()
  set(MFEM_DEBUG OFF)
endif()

# MPI -> hypre; PETSc (optional)
if (MFEM_USE_MPI)
  find_package(MPI REQUIRED)
  set(MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
  # Parallel MFEM depends on hypre
  find_package(HYPRE REQUIRED)
  set(MFEM_HYPRE_VERSION ${HYPRE_VERSION})
  if (MFEM_USE_PETSC)
    find_package(PETSc REQUIRED)
    message(STATUS "Found PETSc version ${PETSC_VERSION}")
    if (PETSC_VERSION AND (PETSC_VERSION VERSION_LESS 3.8.0))
      message(FATAL_ERROR "PETSc version >= 3.8.0 is required")
    endif()
    set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDES})
  endif()
else()
  set(PKGS_NEED_MPI SUPERLU PETSC STRUMPACK PUMI)
  foreach(PKG IN LISTS PKGS_NEED_MPI)
    if (MFEM_USE_${PKG})
      message(STATUS "Disabling package ${PKG} - requires MPI")
      set(MFEM_USE_${PKG} OFF CACHE BOOL "Disabled - requires MPI" FORCE)
    endif()
  endforeach()
endif()

if (MFEM_USE_METIS)
  find_package(METIS REQUIRED)
endif()

# GZSTREAM -> zlib
if (MFEM_USE_GZSTREAM)
  find_package(ZLIB REQUIRED)
endif()

# Backtrace with libunwind
if (MFEM_USE_LIBUNWIND)
  set(MFEMBacktrace_REQUIRED_PACKAGES "Libunwind" "LIBDL" "CXXABIDemangle")
  find_package(MFEMBacktrace REQUIRED)
endif()

# BLAS, LAPACK
if (MFEM_USE_LAPACK)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

# OpenMP
if (MFEM_USE_OPENMP)
  if (MFEM_THREAD_SAFE)
    find_package(OpenMP REQUIRED)
  else()
    message(FATAL_ERROR " *** MFEM_USE_OPENMP requires MFEM_THREAD_SAFE=ON.")
  endif()
endif()

# SuiteSparse (before SUNDIALS which may depend on KLU)
if (MFEM_USE_SUITESPARSE)
  find_package(SuiteSparse REQUIRED
    UMFPACK KLU AMD BTF CHOLMOD COLAMD CAMD CCOLAMD config)
endif()

# SUNDIALS
if (MFEM_USE_SUNDIALS)
  if (NOT MFEM_USE_MPI)
    find_package(SUNDIALS REQUIRED NVector_Serial CVODE ARKODE KINSOL)
  else()
    find_package(SUNDIALS REQUIRED
      NVector_Serial NVector_Parallel NVector_ParHyp CVODE ARKODE KINSOL)
  endif()
endif()

# Mesquite
if (MFEM_USE_MESQUITE)
  find_package(Mesquite REQUIRED)
endif()

# SuperLU_DIST can only be enabled in parallel
if (MFEM_USE_SUPERLU)
  if (MFEM_USE_MPI)
    find_package(SuperLUDist REQUIRED)
  else()
    message(FATAL_ERROR " *** SuperLU_DIST requires that MPI be enabled.")
  endif()
endif()

# STRUMPACK can only be enabled in parallel
if (MFEM_USE_STRUMPACK)
  if (MFEM_USE_MPI)
    find_package(STRUMPACK REQUIRED)
  else()
    message(FATAL_ERROR " *** STRUMPACK requires that MPI be enabled.")
  endif()
endif()

# Gecko
if (MFEM_USE_GECKO)
  find_package(Gecko REQUIRED)
endif()

# GnuTLS
if (MFEM_USE_GNUTLS)
  find_package(_GnuTLS REQUIRED)
endif()

# NetCDF
if (MFEM_USE_NETCDF)
  find_package(NetCDF REQUIRED)
endif()

# MPFR
if (MFEM_USE_MPFR)
  find_package(MPFR REQUIRED)
endif()

if (MFEM_USE_CONDUIT)
    find_package(Conduit REQUIRED conduit relay blueprint )
endif()

# Axom/Sidre
if (MFEM_USE_SIDRE)
  find_package(Axom REQUIRED Sidre SLIC axom_utils)
endif()

# PUMI
if (MFEM_USE_PUMI)
  # If PUMI_DIR was specified, only link to that directory,
  # i.e. don't link to another installation in /usr/lib by mistake
  find_package(SCOREC 2.1.0 REQUIRED OPTIONAL_COMPONENTS gmi_sim
    CONFIG PATHS ${PUMI_DIR} NO_DEFAULT_PATH)
  if (SCOREC_FOUND)
    # Define a header file with the MFEM_USE_SIMMETRIX preprocessor variable
    set(MFEM_USE_SIMMETRIX ${SCOREC_gmi_sim_FOUND})
    set(PUMI_FOUND ${SCOREC_FOUND})
    get_target_property(PUMI_INCLUDE_DIRS
      SCOREC::apf INTERFACE_INCLUDE_DIRECTORIES)
    set(PUMI_LIBRARIES SCOREC::core)
  endif()
endif()

# MFEM_TIMER_TYPE
if (NOT DEFINED MFEM_TIMER_TYPE)
  if (APPLE)
    # use std::clock from <ctime> for UserTime and
    # use mach_absolute_time from <mach/mach_time.h> for RealTime
    set(MFEM_TIMER_TYPE 4)
  elseif (WIN32)
    set(MFEM_TIMER_TYPE 3) # QueryPerformanceCounter from <windows.h>
  else()
    find_package(POSIXClocks)
    if (POSIXCLOCKS_FOUND)
      set(MFEM_TIMER_TYPE 2) # use high-resolution POSIX clocks
    else()
      set(MFEM_TIMER_TYPE 0) # use std::clock from <ctime>
    endif()
  endif()
endif()

# List all possible libraries in order of dependencies.
# [METIS < SuiteSparse]:
#    With newer versions of SuiteSparse which include METIS header using 64-bit
#    integers, the METIS header (with 32-bit indices, as used by mfem) needs to
#    be before SuiteSparse.
set(MFEM_TPLS MPI_CXX OPENMP BLAS LAPACK METIS HYPRE SuiteSparse SUNDIALS PETSC
    MESQUITE SuperLUDist STRUMPACK AXOM CONDUIT GECKO GNUTLS NETCDF MPFR PUMI
    POSIXCLOCKS MFEMBacktrace ZLIB)
# Add all *_FOUND libraries in the variable TPL_LIBRARIES.
set(TPL_LIBRARIES "")
set(TPL_INCLUDE_DIRS "")
foreach(TPL IN LISTS MFEM_TPLS)
  if (${TPL}_FOUND)
    message(STATUS "MFEM: using package ${TPL}")
    list(APPEND TPL_LIBRARIES ${${TPL}_LIBRARIES})
    list(APPEND TPL_INCLUDE_DIRS ${${TPL}_INCLUDE_DIRS})
  endif()
endforeach(TPL)
list(REMOVE_DUPLICATES TPL_LIBRARIES)
list(REMOVE_DUPLICATES TPL_INCLUDE_DIRS)
# message(STATUS "TPL_INCLUDE_DIRS = ${TPL_INCLUDE_DIRS}")
include_directories(${TPL_INCLUDE_DIRS})

if (OPENMP_FOUND)
  message(STATUS "MFEM: using package OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message(STATUS "MFEM build type: CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "MFEM version: v${MFEM_VERSION_STRING}")
message(STATUS "MFEM git string: ${MFEM_GIT_STRING}")

# Windows specific
set(_USE_MATH_DEFINES ${WIN32})

#-------------------------------------------------------------------------------
# Define and configure the MFEM library
#-------------------------------------------------------------------------------

# Headers and sources
set(SOURCES "")
set(HEADERS "")
set(MFEM_SOURCE_DIRS general linalg mesh fem)
foreach(DIR IN LISTS MFEM_SOURCE_DIRS)
  add_subdirectory(${DIR})
endforeach()
add_subdirectory(config)
set(MASTER_HEADERS
  ${PROJECT_SOURCE_DIR}/mfem.hpp
  ${PROJECT_SOURCE_DIR}/mfem-performance.hpp)

set(_lib_path "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL "")
set(CMAKE_INSTALL_RPATH "${_lib_path}" CACHE PATH "")
set(CMAKE_INSTALL_NAME_DIR "${_lib_path}" CACHE PATH "")

# Declaring the library
add_library(mfem ${SOURCES} ${HEADERS} ${MASTER_HEADERS})
# message(STATUS "TPL_LIBRARIES = ${TPL_LIBRARIES}")
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
  target_link_libraries(mfem PUBLIC ${TPL_LIBRARIES})
else()
  target_link_libraries(mfem ${TPL_LIBRARIES})
endif()
if (MINGW)
  target_link_libraries(mfem ws2_32)
endif()
set_target_properties(mfem PROPERTIES VERSION "${mfem_VERSION}")
set_target_properties(mfem PROPERTIES SOVERSION "${mfem_VERSION}")

# If building out-of-source, define MFEM_BUILD_DIR to point to the build
# directory.
if (NOT ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}"))
  target_compile_definitions(mfem PRIVATE
    "MFEM_BUILD_DIR=${PROJECT_BINARY_DIR}")
endif()

# Generate configuration file in the build directory: config/_config.hpp.
configure_file(
  "${PROJECT_SOURCE_DIR}/config/cmake/config.hpp.in"
  "${PROJECT_BINARY_DIR}/config/_config.hpp")

# Create substitute mfem.hpp and mfem-performance.hpp in the build directory,
# if it is different from the source directory.
if (NOT ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}"))
  foreach(Header mfem.hpp mfem-performance.hpp)
    message(STATUS
      "Writing substitute header --> \"${Header}\"")
    file(WRITE "${PROJECT_BINARY_DIR}/${Header}"
"// Auto-generated file.
#define MFEM_BUILD_DIR ${PROJECT_BINARY_DIR}
#include \"${PROJECT_SOURCE_DIR}/${Header}\"
")
    # This version will be installed in the top include directory:
    file(WRITE "${PROJECT_BINARY_DIR}/InstallHeaders/${Header}"
"// Auto-generated file.
#include \"mfem/${Header}\"
")
  endforeach()
endif()

#-------------------------------------------------------------------------------
# Examples, miniapps, and testing
#-------------------------------------------------------------------------------

# Enable testing if required
if (MFEM_ENABLE_TESTING)
  enable_testing()
endif()

# Define a target that all examples and miniapps will depend on.
set(MFEM_EXEC_PREREQUISITES_TARGET_NAME exec_prerequisites)
add_custom_target(${MFEM_EXEC_PREREQUISITES_TARGET_NAME})

# Create a target for all examples and, optionally, enable it.
set(MFEM_ALL_EXAMPLES_TARGET_NAME examples)
add_mfem_target(${MFEM_ALL_EXAMPLES_TARGET_NAME} ${MFEM_ENABLE_EXAMPLES})
add_subdirectory(examples EXCLUDE_FROM_ALL)

# Create a target for all miniapps and, optionally, enable it.
set(MFEM_ALL_MINIAPPS_TARGET_NAME miniapps)
add_mfem_target(${MFEM_ALL_MINIAPPS_TARGET_NAME} ${MFEM_ENABLE_MINIAPPS})
add_subdirectory(miniapps EXCLUDE_FROM_ALL)

# Target to build all executables, i.e. everything.
add_custom_target(exec)
add_dependencies(exec
  ${MFEM_ALL_EXAMPLES_TARGET_NAME} ${MFEM_ALL_MINIAPPS_TARGET_NAME})
# Here, we want to "add_dependencies(test exec)". However, dependencies for
# 'test' (and other built-in targets) can not be added with add_dependencies():
#  - https://gitlab.kitware.com/cmake/cmake/issues/8438
#  - https://cmake.org/Bug/view.php?id=8438

# Add a target to copy the mfem data directory to the build directory
add_custom_command(OUTPUT data_is_copied
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/data data
  COMMAND ${CMAKE_COMMAND} -E touch data_is_copied
  COMMENT "Copying the data directory ...")
add_custom_target(copy_data DEPENDS data_is_copied)
# Add 'copy_data' as a prerequisite for all executables, if the source and the
# build directories are not the same.
if (NOT ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}"))
  add_dependencies(${MFEM_EXEC_PREREQUISITES_TARGET_NAME} copy_data)
endif()

# Add 'check' target - quick test
if (NOT MFEM_USE_MPI)
  add_custom_target(check
    ${CMAKE_CTEST_COMMAND} -R '^ex1_ser' -C ${CMAKE_CFG_INTDIR}
    USES_TERMINAL)
  add_dependencies(check ex1)
else()
  add_custom_target(check
    ${CMAKE_CTEST_COMMAND} -R '^ex1p' -C ${CMAKE_CFG_INTDIR}
    USES_TERMINAL)
  add_dependencies(check ex1p)
endif()

#-------------------------------------------------------------------------------
# Documentation
#-------------------------------------------------------------------------------
add_subdirectory(doc)

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
set(INSTALL_INCLUDE_DIR include
  CACHE PATH "Relative path for installing header files.")
set(INSTALL_LIB_DIR lib
  CACHE PATH "Relative path for installing the library.")
# other options: "share/mfem/cmake", "lib/mfem/cmake"
set(INSTALL_CMAKE_DIR lib/cmake/mfem
  CACHE PATH "Relative path for installing cmake config files.")

# The 'install' target will not depend on 'all'.
# set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME Development)

# Install the library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME_UC}Targets
  DESTINATION ${INSTALL_LIB_DIR})

# Install the master headers
foreach(Header mfem.hpp mfem-performance.hpp)
  install(FILES ${PROJECT_BINARY_DIR}/InstallHeaders/${Header}
    DESTINATION ${INSTALL_INCLUDE_DIR})
endforeach()
install(FILES ${MASTER_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR}/mfem)

# Install the headers; currently, the miniapps headers are excluded
install(DIRECTORY ${MFEM_SOURCE_DIRS}
  DESTINATION ${INSTALL_INCLUDE_DIR}/mfem
  FILES_MATCHING PATTERN "*.hpp")

# Install ${HEADERS}
# ---
# foreach (HDR ${HEADERS})
#   file(RELATIVE_PATH REL_HDR ${PROJECT_SOURCE_DIR} ${HDR})
#   get_filename_component(DIR ${REL_HDR} PATH)
#   install(FILES ${REL_HDR} DESTINATION ${INSTALL_INCLUDE_DIR}/${DIR})
# endforeach()

# Install the configuration header files
install(FILES ${PROJECT_BINARY_DIR}/config/_config.hpp
  DESTINATION ${INSTALL_INCLUDE_DIR}/mfem/config
  RENAME config.hpp)

install(FILES ${PROJECT_SOURCE_DIR}/config/tconfig.hpp
  DESTINATION ${INSTALL_INCLUDE_DIR}/mfem/config)

# Package the whole thing up nicely
include(CMakePackageConfigHelpers)

# Add all targets to the build-tree export set
export(TARGETS ${PROJECT_NAME}
  FILE "${PROJECT_BINARY_DIR}/MFEMTargets.cmake")

# Export the package for use from the build-tree (this registers the build-tree
# with the CMake user package registry.)
# TODO: How do we register the install-tree? Replacing the build-tree?
export(PACKAGE ${PROJECT_NAME})

# Extract the include directories required to use MFEM
get_target_property(MFEM_TPL_INCLUDE_DIRS mfem INCLUDE_DIRECTORIES)
if (NOT MFEM_TPL_INCLUDE_DIRS)
  set(MFEM_TPL_INCLUDE_DIRS "")
endif()

# This is the build-tree version
set(INCLUDE_INSTALL_DIRS ${PROJECT_BINARY_DIR} ${MFEM_TPL_INCLUDE_DIRS})
set(LIB_INSTALL_DIR ${PROJECT_BINARY_DIR})
configure_package_config_file(config/cmake/MFEMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MFEMConfig.cmake
  INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# This is the version that will be installed
set(INCLUDE_INSTALL_DIRS ${INSTALL_INCLUDE_DIR}  ${MFEM_TPL_INCLUDE_DIRS})
set(LIB_INSTALL_DIR ${INSTALL_LIB_DIR})
configure_package_config_file(config/cmake/MFEMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/MFEMConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKE_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Write the version file (same for build and install tree)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MFEMConfigVersion.cmake
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY SameMajorVersion )

# Install the config files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/MFEMConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MFEMConfigVersion.cmake
  DESTINATION ${INSTALL_CMAKE_DIR})

# Install the export set for use with the install-tree
install(EXPORT ${PROJECT_NAME_UC}Targets
    DESTINATION ${INSTALL_CMAKE_DIR})
