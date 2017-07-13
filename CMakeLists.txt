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
project(mfem CXX)
# Current version of MFEM, see also `makefile`.
#   mfem_VERSION = (string)
#   MFEM_VERSION = (int)   [automatically derived from mfem_VERSION]
set(${PROJECT_NAME}_VERSION 3.3.1)

# Prohibit in-source build
if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(FATAL_ERROR
    "MFEM does not support in-source CMake builds at this time.")
endif (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})

if (CMAKE_VERSION VERSION_LESS 3.2 OR MFEM_USE_SIDRE OR MFEM_USE_PETSC)
   # This seems to be needed by:
   #  * find_package(BLAS REQUIRED) and
   #  * find_package(HDF5 REQUIRED) needed, in turn, by:
   #    - find_package(AXOM REQUIRED)
   #  * find_package(PETSc REQUIRED)
   enable_language(C)
endif()

# CMake needs to know where to find things
set(MFEM_CMAKE_PATH ${PROJECT_SOURCE_DIR}/config)
set(CMAKE_MODULE_PATH ${MFEM_CMAKE_PATH}/cmake/modules)

# Load MFEM CMake utilities.
include(MfemCmakeUtilities)

string(TOUPPER "${PROJECT_NAME}" PROJECT_NAME_UC)
mfem_version_to_int(${${PROJECT_NAME}_VERSION} ${PROJECT_NAME_UC}_VERSION)

#-------------------------------------------------------------------------------
# Process configuration options
#-------------------------------------------------------------------------------

# MFEM_DEBUG
if (CMAKE_BUILD_TYPE MATCHES "Debug|debug|DEBUG")
  set(MFEM_DEBUG ON)
else()
  set(MFEM_DEBUG OFF)
endif()

# MPI -> hypre, METIS
if (MFEM_USE_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  # Parallel MFEM depends on hypre and METIS
  find_package(HYPRE REQUIRED)
  set(MFEM_HYPRE_VERSION ${HYPRE_VERSION})
  include_directories(${HYPRE_INCLUDE_DIRS})
  find_package(METIS REQUIRED)
  include_directories(${METIS_INCLUDE_DIRS})
  if (MFEM_USE_PETSC)
    find_package(PETSc REQUIRED)
    message(STATUS "Found PETSc version ${PETSC_VERSION}")
    if (PETSC_VERSION AND (PETSC_VERSION VERSION_LESS 3.7.5.99))
      message(FATAL_ERROR "PETSc version >= 3.7.5.99 is required")
    endif()
    include_directories(${PETSC_INCLUDES})
  endif()
else()
  set(PKGS_NEED_MPI SUPERLU PETSC STRUMPACK)
  foreach(PKG IN LISTS PKGS_NEED_MPI)
    if (MFEM_USE_${PKG})
      message(STATUS "Disabling package ${PKG} - requires MPI")
      set(MFEM_USE_${PKG} OFF CACHE BOOL "Disabled - requires MPI" FORCE)
    endif()
  endforeach()
endif()

# GZSTREAM -> zlib
if (MFEM_USE_GZSTREAM)
  find_package(ZLIB REQUIRED)
  include_directories(${ZLIB_INCLUDE_DIRS})
endif()

# Backtrace with libunwind
if (MFEM_USE_LIBUNWIND)
  set(MFEMBacktrace_REQUIRED_PACKAGES "Libunwind" "LIBDL" "CXXABIDemangle")
  find_package(MFEMBacktrace REQUIRED)
  include_directories(${LIBUNWIND_INCLUDE_DIRS})
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
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif()

# SUNDIALS
if (MFEM_USE_SUNDIALS)
  if (NOT MFEM_USE_MPI)
    find_package(SUNDIALS REQUIRED NVector_Serial CVODE ARKODE KINSOL)
  else()
    find_package(SUNDIALS REQUIRED
      NVector_Serial NVector_Parallel NVector_ParHyp CVODE ARKODE KINSOL)
  endif()
  include_directories(${SUNDIALS_INCLUDE_DIRS})
endif()

# Mesquite
if (MFEM_USE_MESQUITE)
  find_package(Mesquite REQUIRED)
  include_directories(${MESQUITE_INCLUDE_DIRS})
endif()

# SuperLU_DIST can only be enabled in parallel
if (MFEM_USE_SUPERLU)
  if (MFEM_USE_MPI)
    find_package(SuperLUDist REQUIRED)
    include_directories(${SuperLUDist_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR " *** SuperLU_DIST requires that MPI be enabled.")
  endif()
endif()

# STRUMPACK can only be enabled in parallel
if (MFEM_USE_STRUMPACK)
  if (MFEM_USE_MPI)
    find_package(STRUMPACK REQUIRED)
    include_directories(${STRUMPACK_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR " *** STRUMPACK requires that MPI be enabled.")
  endif()
endif()

# Gecko
if (MFEM_USE_GECKO)
  find_package(Gecko REQUIRED)
  include_directories(${GECKO_INCLUDE_DIRS})
endif()

# GnuTLS
if (MFEM_USE_GNUTLS)
  find_package(_GnuTLS REQUIRED)
  include_directories(${GNUTLS_INCLUDE_DIRS})
endif()

# NetCDF
if (MFEM_USE_NETCDF)
  find_package(NetCDF REQUIRED)
  include_directories(${NETCDF_INCLUDE_DIRS})
endif()

# MPFR
if (MFEM_USE_MPFR)
  find_package(MPFR REQUIRED)
  include_directories(${MPFR_INCLUDE_DIRS})
endif()

# Axom/Sidre
if (MFEM_USE_SIDRE)
  if (NOT MFEM_USE_MPI)
    find_package(Axom REQUIRED Sidre SLIC axom_utils)
  else()
    find_package(Axom REQUIRED Sidre SPIO SLIC axom_utils)
  endif()
  include_directories(${AXOM_INCLUDE_DIRS})
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
set(MFEM_TPLS MPI_CXX HYPRE OPENMP SUNDIALS MESQUITE SuiteSparse SuperLUDist
    ParMETIS METIS LAPACK BLAS GECKO GNUTLS NETCDF PETSC MPFR AXOM POSIXCLOCKS
    MFEMBacktrace ZLIB STRUMPACK)
# Add all *_FOUND libraries in the variable TPL_LIBRARIES.
set(TPL_LIBRARIES "")
foreach(TPL IN LISTS MFEM_TPLS)
  if (${TPL}_FOUND)
    message(STATUS "MFEM: using package ${TPL}")
    list(APPEND TPL_LIBRARIES ${${TPL}_LIBRARIES})
  endif()
endforeach(TPL)

if (OPENMP_FOUND)
  message(STATUS "MFEM: using package OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message(STATUS "MFEM build type: CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Windows specific
set(_USE_MATH_DEFINES ${WIN32})

#-------------------------------------------------------------------------------
# Define and configure the MFEM library
#-------------------------------------------------------------------------------

list(APPEND CMAKE_CXX_IGNORE_EXTENSIONS tpp)

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

# Declaring the library
add_library(mfem ${SOURCES} ${HEADERS} ${MASTER_HEADERS})
list(REMOVE_DUPLICATES TPL_LIBRARIES)
# message(STATUS " TPL_LIBRARIES = ${TPL_LIBRARIES}")
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
  target_link_libraries(mfem PUBLIC ${TPL_LIBRARIES})
else()
  target_link_libraries(mfem ${TPL_LIBRARIES})
endif()
if (MINGW)
  target_link_libraries(mfem ws2_32)
endif()

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
    ${CMAKE_CTEST_COMMAND} -R ex1_ser -E performance -C ${CMAKE_CFG_INTDIR}
    USES_TERMINAL)
  add_dependencies(check ex1)
else()
  add_custom_target(check
    ${CMAKE_CTEST_COMMAND} -R ex1p -E performance -C ${CMAKE_CFG_INTDIR}
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
install(FILES ${MASTER_HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR})

# Install the headers; currently, the miniapps headers are excluded
install(DIRECTORY ${MFEM_SOURCE_DIRS}
  DESTINATION ${INSTALL_INCLUDE_DIR}
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
  DESTINATION ${INSTALL_INCLUDE_DIR}/config
  RENAME config.hpp)

install(FILES ${PROJECT_SOURCE_DIR}/config/tconfig.hpp
  DESTINATION ${INSTALL_INCLUDE_DIR}/config)

# Package the whole thing up nicely
include(CMakePackageConfigHelpers)

# Add all targets to the build-tree export set
export(TARGETS ${PROJECT_NAME}
  FILE "${PROJECT_BINARY_DIR}/MFEMTargets.cmake")

# Export the package for use from the build-tree (this registers the build-tree
# with the CMake user package registry.)
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
