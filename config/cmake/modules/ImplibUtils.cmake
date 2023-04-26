#[=======================================================================[.rst:
IMPLIB_UTILS
------------

Tools for CMake on WIN32 to associate IMPORTED_IMPLIB paths (as discovered
by the :command:`find_library` command) with their IMPORTED_LOCATION DLLs.

Writing Find modules that create ``SHARED IMPORTED`` targets with the
correct ``IMPORTED_IMPLIB`` and ``IMPORTED_LOCATION`` properties is a
requirement for ``$<TARGET_RUNTIME_DLLS>`` to work correctly. (Probably
``IMPORTED_RUNTIME_DEPENDENCIES`` as well.)

Macros Provided
^^^^^^^^^^^^^^^

Currently the only tool here is ``implib_to_dll``. It takes a single
argument, the __name__ (_not_ value!) of a prefixed ``<prefix>_IMPLIB``
variable (containing the path to a ``.lib`` or ``.dll.a`` import library).

``implib_to_dll`` will attempt to locate the corresponding ``.dll`` file
for that import library, and set the cache variable ``<prefix>_LIBRARY``
to its location.

``implib_to_dll`` relies on the ``dlltool.exe`` utility. The path can
be set by defining ``DLLTOOL_EXECUTABLE`` in the cache prior to
including this module, if it is not set implib_utils will attempt to locate
``dlltool.exe`` using ``find_program()``.

Revision history
^^^^^^^^^^^^^^^^
2021-10-14 - Initial version

Author: FeRD (Frank Dana) <ferdnyc@gmail.com>
License: CC0-1.0 (Creative Commons Universal Public Domain Dedication)
#]=======================================================================]
include_guard(DIRECTORY)

if (NOT WIN32)
  # Nothing to do here!
  return()
endif()

if (NOT DEFINED DLLTOOL_EXECUTABLE)
  find_program(DLLTOOL_EXECUTABLE
    NAMES dlltool dlltool.exe
    DOC "The path to the DLLTOOL utility"
  )
  if (DLLTOOL_EXECUTABLE STREQUAL "DLLTOOL_EXECUTABLE-NOTFOUND")
    message(WARNING "DLLTOOL not available, cannot continue")
    return()
  endif()
  message(DEBUG "Found dlltool at ${DLLTOOL_EXECUTABLE}")
endif()

#
### Macro: implib_to_dll
#
# (Win32 only)
# Uses dlltool.exe to find the name of the dll associated with the
# supplied import library.
macro(implib_to_dll _implib_var)
  set(_implib ${${_implib_var}})
  set(_library_var "${_implib_var}")
  # Automatically update the name, assuming it's in the correct format
  string(REGEX REPLACE
    [[_IMPLIBS$]] [[_LIBRARIES]]
    _library_var "${_library_var}")
  string(REGEX REPLACE
    [[_IMPLIB$]] [[_LIBRARY]]
    _library_var "${_library_var}")
  # We can't use the input variable name without blowing away the
  # previously-discovered contents, so that's a non-starter
  if ("${_implib_var}" STREQUAL "${_library_var}")
    message(ERROR "Name collision! You probably didn't pass"
    "implib_to_dll() a correctly-formatted variable name."
    "Only <prefix>_IMPLIB or <prefix>_IMPLIBS is supported.")
    return()
  endif()

  if(EXISTS "${_implib}")
    message(DEBUG "Looking up dll name for import library ${_implib}")

    # Check the directory where the import lib is found
    get_filename_component(_implib_dir ".." REALPATH
                           BASE_DIR "${_implib}")
    message(DEBUG "Checking import lib directory ${_implib_dir}")

    # Add a check in ../../bin/, relative to the import library
    get_filename_component(_bindir "../../bin" REALPATH
                           BASE_DIR "${_implib}")
    message(DEBUG "Also checking ${_bindir}")

    execute_process(COMMAND
      "${DLLTOOL_EXECUTABLE}" -I "${_implib}"
      OUTPUT_VARIABLE _dll_name
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(DEBUG "DLLTOOL returned ${_dll_name}")

    find_program(${_library_var}
      NAMES ${_dll_name}
      HINTS
        ${_bindir}
        ${_implib_dir}
      PATHS
        ENV PATH
    )
    set(${_library_var} "${${_library_var}}" PARENT_SCOPE)
    message(DEBUG "Set ${_library_var} to ${${_library_var}}")
  endif()
endmacro() 
