# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

#[=======================================================================[.rst:
FindLIBBACKTRACE
-------

Finds the LIBBACKTRACE library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``LIBBACKTRACE_FOUND``
  True if the system has the LIBBACKTRACE library.
``LIBBACKTRACE_INCLUDE_DIRS``
  Include directories needed to use LIBBACKTRACE.
``LIBBACKTRACE_LIBRARIES``
  Libraries needed to link to LIBBACKTRACE.
#]=======================================================================]

include(MfemCmakeUtilities)
include(FindPackageHandleStandardArgs)

# https://github.com/ianlancetaylor/libbacktrace

if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin") ################################
  # Find libbacktrace.
  # Defines the following variables:
  #   - LIBBACKTRACE_FOUND
  #   - LIBBACKTRACE_LIBRARIES    (if needed)
  #   - LIBBACKTRACE_INCLUDE_DIRS (if needed)
  message(STATUS "[🟠 LIBBACKTRACE 🟠] Looking for LIBBACKTRACE ...")

  set(BACKTRACE_DIR "$ENV{HOME}/usr/local/backtrace" CACHE PATH "")

  # set(LIBBACKTRACE_SKIP_LOOKING_MSG TRUE)
  mfem_find_package(LIBBACKTRACE LIBBACKTRACE BACKTRACE_DIR 
                    "include" backtrace.h
                    "lib" backtrace
                    "Paths to headers required by LIBBACKTRACE." 
                    "Libraries required by LIBBACKTRACE.")

elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux") #############################
  find_package(LIBBACKTRACE QUIET NO_MODULE)
  # set(LIBBACKTRACE_FOUND TRUE)  
  ## Find headers and libraries
  # should be in CMAKE_PREFIX_PATH if libunwind is
  # loaded with spack.
  find_library(LIBBACKTRACE_LIBRARIES libbacktrace.so)
  find_path(LIBBACKTRACE_INCLUDE_DIRS backtrace.h)
  if(NOT LIBBACKTRACE_LIBRARIES OR NOT LIBBACKTRACE_INCLUDE_DIRS)
      set(LIBBACKTRACE_FOUND FALSE)
  endif()
  find_package_handle_standard_args(LIBBACKTRACE DEFAULT_MSG
      LIBBACKTRACE_FOUND
      LIBBACKTRACE_LIBRARIES
      LIBBACKTRACE_INCLUDE_DIRS)
else() 
  message(STATUS "[🟠 LIBBACKTRACE 🟠] Unsupported platform!")
endif()

message(STATUS "[🟠 LIBBACKTRACE 🟠] LIBBACKTRACE_FOUND: ${LIBBACKTRACE_FOUND}")
message(STATUS "[🟠 LIBBACKTRACE 🟠] LIBBACKTRACE_INCLUDE_DIRS: ${LIBBACKTRACE_INCLUDE_DIRS}")
message(STATUS "[🟠 LIBBACKTRACE 🟠] LIBBACKTRACE_LIBRARIES: ${LIBBACKTRACE_LIBRARIES}")
message(STATUS "[🟠 LIBBACKTRACE 🟠] LIBBACKTRACE_FIND_VERSION: ${LIBBACKTRACE_FIND_VERSION}")
message(STATUS "[🟠 LIBBACKTRACE 🟠] LIBBACKTRACE_FIND_COMPONENTS: ${LIBBACKTRACE_FIND_COMPONENTS}")

if (LIBBACKTRACE_FOUND)
  add_library(LIBBACKTRACE::LIBBACKTRACE UNKNOWN IMPORTED)
  set_target_properties(LIBBACKTRACE::LIBBACKTRACE PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LIBBACKTRACE_INCLUDE_DIRS}"
      IMPORTED_LOCATION ${LIBBACKTRACE_LIBRARIES})
endif()

mark_as_advanced(LIBBACKTRACE_LIBRARIES LIBBACKTRACE_INCLUDE_DIRS LIBBACKTRACE_FOUND)
