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

find_package(LIBBACKTRACE QUIET NO_MODULE)
if(LIBBACKTRACE_FOUND)
    message("LIBBACKTRACE ${LIBBACKTRACE_FIND_VERSION} found.")
    if(LIBBACKTRACE_FIND_COMPONENTS)
        message("LIBBACKTRACE components found:")
        message("${LIBBACKTRACE_FIND_COMPONENTS}")
    endif()
    return()
endif()

set(LIBBACKTRACE_FOUND TRUE)

## Find headers and libraries
# should be in CMAKE_PREFIX_PATH if libunwind is
# loaded with spack.
find_library(LIBBACKTRACE_LIBRARIES libbacktrace.so)
find_path(LIBBACKTRACE_INCLUDE_DIRS backtrace.h)
if(NOT LIBBACKTRACE_LIBRARIES OR NOT LIBBACKTRACE_INCLUDE_DIRS)
    set(LIBBACKTRACE_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBBACKTRACE DEFAULT_MSG
                                  LIBBACKTRACE_FOUND
                                  LIBBACKTRACE_LIBRARIES
                                  LIBBACKTRACE_INCLUDE_DIRS)

if (LIBBACKTRACE_FOUND)
   add_library(LIBBACKTRACE::LIBBACKTRACE UNKNOWN IMPORTED)
   set_target_properties(LIBBACKTRACE::LIBBACKTRACE PROPERTIES
       INTERFACE_INCLUDE_DIRECTORIES "${LIBBACKTRACE_INCLUDE_DIRS}"
       IMPORTED_LOCATION ${LIBBACKTRACE_LIBRARIES}
       )
endif()

mark_as_advanced(LIBBACKTRACE_LIBRARIES LIBBACKTRACE_INCLUDE_DIRS LIBBACKTRACE_FOUND)
