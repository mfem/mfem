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

# Function that converts a version string of the form 'major[.minor[.patch]]' to
# the integer ((major * 100) + minor) * 100 + patch.
function(mfem_version_to_int VersionString VersionIntVar)
  if ("${VersionString}" MATCHES "^([0-9]+)(.*)$")
    set(Major "${CMAKE_MATCH_1}")
    set(MinorPatchString "${CMAKE_MATCH_2}")
  else()
    set(Major 0)
  endif()
  if ("${MinorPatchString}" MATCHES "^\\.([0-9]+)(.*)$")
    set(Minor "${CMAKE_MATCH_1}")
    set(PatchString "${CMAKE_MATCH_2}")
  else()
    set(Minor 0)
  endif()
  if ("${PatchString}" MATCHES "^\\.([0-9]+)(.*)$")
    set(Patch "${CMAKE_MATCH_1}")
  else()
    set(Patch 0)
  endif()
  math(EXPR VersionInt "(${Major}*100+${Minor})*100+${Patch}")
  set(${VersionIntVar} ${VersionInt} PARENT_SCOPE)
endfunction()

# A handy function to add the current source directory to a local
# filename. To be used for creating a list of sources.
function(convert_filenames_to_full_paths NAMES)
  unset(tmp_names)
  foreach(name ${${NAMES}})
    list(APPEND tmp_names ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  endforeach()
  set(${NAMES} ${tmp_names} PARENT_SCOPE)
endfunction()

# Simple shortcut to add_custom_target() with option to add the target to the
# main target.
function(add_mfem_target TARGET_NAME ADD_TO_ALL)
  if (ADD_TO_ALL)
    # add TARGET_NAME to the main target
    add_custom_target(${TARGET_NAME} ALL)
  else()
    # do not add TARGET_NAME to the main target
    add_custom_target(${TARGET_NAME})
  endif()
endfunction()

# Add mfem examples
function(add_mfem_examples EXE_SRCS)
  set(EXE_PREFIX "")
  set(EXE_PREREQUISITE "")
  set(EXE_NEEDED_BY "")
  if (${ARGC} GREATER 1)
    set(EXE_PREFIX "${ARGV1}")
    if (${ARGC} GREATER 2)
      set(EXE_PREREQUISITE "${ARGV2}")
      if (${ARGC} GREATER 3)
        set(EXE_NEEDED_BY "${ARGV3}")
      endif()
    endif()
  endif()
  foreach(SRC_FILE IN LISTS ${EXE_SRCS})
    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)

    string(REPLACE ".cpp" "" EXE_NAME "${EXE_PREFIX}${SRC_FILENAME}")
    add_executable(${EXE_NAME} ${SRC_FILE})
    # If given a prefix, don't add the example to the list of examples to build.
    if (NOT EXE_PREFIX)
      add_dependencies(${MFEM_ALL_EXAMPLES_TARGET_NAME} ${EXE_NAME})
    elseif (EXE_NEEDED_BY)
      add_dependencies(${EXE_NEEDED_BY} ${EXE_NAME})
    endif()
    add_dependencies(${EXE_NAME}
      ${MFEM_EXEC_PREREQUISITES_TARGET_NAME} ${EXE_PREREQUISITE})

    target_link_libraries(${EXE_NAME} mfem)
    if (MFEM_USE_MPI)
      # Not needed: (mfem already links with MPI_CXX_LIBRARIES)
      # target_link_libraries(${EXE_NAME} ${MPI_CXX_LIBRARIES})

      # Language-specific include directories:
      if (MPI_CXX_INCLUDE_PATH)
        target_include_directories(${EXE_NAME} PRIVATE "${MPI_CXX_INCLUDE_PATH}")
      endif()
      if (MPI_CXX_COMPILE_FLAGS)
        separate_arguments(MPI_CXX_COMPILE_ARGS UNIX_COMMAND
          "${MPI_CXX_COMPILE_FLAGS}")
        target_compile_options(${EXE_NAME} PRIVATE ${MPI_CXX_COMPILE_ARGS})
      endif()

      if (MPI_CXX_LINK_FLAGS)
        set_target_properties(${EXE_NAME} PROPERTIES
          LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
      endif()
    endif()
  endforeach(SRC_FILE)
endfunction()

# A slightly more versatile function for adding miniapps to MFEM
function(add_mfem_miniapp MFEM_EXE_NAME)
  # Parse the input arguments looking for the things we need
  set(POSSIBLE_ARGS "MAIN" "EXTRA_SOURCES" "EXTRA_HEADERS" "EXTRA_OPTIONS" "EXTRA_DEFINES" "LIBRARIES")
  set(CURRENT_ARG)
  foreach(arg ${ARGN})
    list(FIND POSSIBLE_ARGS ${arg} is_arg_name)
    if (${is_arg_name} GREATER -1)
      set(CURRENT_ARG ${arg})
      set(${CURRENT_ARG}_LIST)
    else()
      list(APPEND ${CURRENT_ARG}_LIST ${arg})
    endif()
  endforeach()

  # Actually add the executable
  add_executable(${MFEM_EXE_NAME} ${MAIN_LIST}
    ${EXTRA_SOURCES_LIST} ${EXTRA_HEADERS_LIST})
  add_dependencies(${MFEM_ALL_MINIAPPS_TARGET_NAME} ${MFEM_EXE_NAME})
  add_dependencies(${MFEM_EXE_NAME} ${MFEM_EXEC_PREREQUISITES_TARGET_NAME})

  # Append the additional libraries and options
  if (LIBRARIES_LIST)
    if(CMAKE_VERSION VERSION_GREATER 2.8.11)
      target_link_libraries(${MFEM_EXE_NAME} PRIVATE ${LIBRARIES_LIST})
    else()
      target_link_libraries(${MFEM_EXE_NAME} ${LIBRARIES_LIST})
    endif()
  endif()
  if (EXTRA_OPTIONS_LIST)
    string(REPLACE ";" " " EXTRA_OPTIONS_STRING "${EXTRA_OPTIONS_LIST}")
    message(STATUS "${MFEM_EXE_NAME}: add flags \"${EXTRA_OPTIONS_STRING}\"")
    if(CMAKE_VERSION VERSION_GREATER 2.8.11)
      target_compile_options(${MFEM_EXE_NAME} PRIVATE ${EXTRA_OPTIONS_LIST})
    else()
      get_target_property(THIS_COMPILE_FLAGS ${MFEM_EXE_NAME} COMPILE_FLAGS)
      if (THIS_COMPILE_FLAGS)
        set(THIS_COMPILE_FLAGS "${THIS_COMPILE_FLAGS} ${EXTRA_OPTIONS_STRING}")
      else()
        set(THIS_COMPILE_FLAGS "${EXTRA_OPTIONS_STRING}")
      endif()
      set_target_properties(${MFEM_EXE_NAME}
        PROPERTIES COMPILE_FLAGS ${THIS_COMPILE_FLAGS})
    endif()
  endif()
  if (EXTRA_DEFINES_LIST)
    target_compile_definitions(${MFEM_EXE_NAME} PRIVATE ${EXTRA_DEFINES_LIST})
  endif()

  # Handle the MPI separately
  if (MFEM_USE_MPI)
    # Add MPI_CXX_LIBRARIES, in case this target does not link with mfem.
    if(CMAKE_VERSION VERSION_GREATER 2.8.11)
      target_link_libraries(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
    else()
      target_link_libraries(${MFEM_EXE_NAME} ${MPI_CXX_LIBRARIES})
    endif()

    if (MPI_CXX_INCLUDE_PATH)
      target_include_directories(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    endif()
    if (MPI_CXX_COMPILE_FLAGS)
      target_compile_options(${MFEM_EXE_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()

    if (MPI_CXX_LINK_FLAGS)
      set_target_properties(${MFEM_EXE_NAME} PROPERTIES
        LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
    endif()
  endif()
endfunction()


# Auxiliary function, used in mfem_find_package().
function(mfem_find_component Prefix DirVar IncSuffixes Header LibSuffixes Lib
         IncDoc LibDoc)

  if (Lib)
    if (${DirVar} OR EnvDirVar)
      find_library(${Prefix}_LIBRARIES ${Lib}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${LibSuffixes}
        NO_DEFAULT_PATH
        DOC "${LibDoc}")
    endif()
    find_library(${Prefix}_LIBRARIES ${Lib}
      PATH_SUFFIXES ${LibSuffixes}
      DOC "${LibDoc}")
  endif()

  if (Header)
    if (${DirVar} OR EnvDirVar)
      find_path(${Prefix}_INCLUDE_DIRS ${Header}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${IncSuffixes}
        NO_DEFAULT_PATH
        DOC "${IncDoc}")
    endif()
    find_path(${Prefix}_INCLUDE_DIRS ${Header}
      PATH_SUFFIXES ${IncSuffixes}
      DOC "${IncDoc}")
  endif()

endfunction(mfem_find_component)


#   MFEM version of find_package that searches for header/library and if
#   successful, optionally checks building (compile + link) one or more given
#   code snippets. Additionally, a list of required/optional/alternative
#   packages (given by ${Name}_REQUIRED_PACKAGES) are searched for and added to
#   the ${Prefix}_INCLUDE_DIRS and ${Prefix}_LIBRARIES lists. The function
#   defines the following CACHE variables:
#
#      ${Prefix}_FOUND
#      ${Prefix}_INCLUDE_DIRS
#      ${Prefix}_LIBRARIES
#
#   If ${Name}_SKIP_LOOKING_MSG is true, skip the initial "Looking ..." message.
#
#   This function is intended to be called from the script Find${Name}.cmake
#
function(mfem_find_package Name Prefix DirVar IncSuffixes Header LibSuffixes
         Lib IncDoc LibDoc)

  # Quick return
  if (${Prefix}_FOUND)
    return()
  elseif (${Prefix}_INCLUDE_DIRS OR ${Prefix}_LIBRARIES)
    # If ${Prefix}_INCLUDE_DIRS or ${Prefix}_LIBRARIES are defined, accept them
    # silently.
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)
    return()
  endif()

  set(EnvDirVar "$ENV{${DirVar}}")
  if (NOT ${Name}_FIND_QUIETLY)
    if (NOT ${Name}_SKIP_LOOKING_MSG)
      message(STATUS "Looking for ${Name} ...")
    endif()
    if (${DirVar})
      message(STATUS "   in ${DirVar} = ${${DirVar}}")
    endif()
    if (EnvDirVar)
      message(STATUS "   in ENV{${DirVar}} = ${EnvDirVar}")
    endif()
  endif()

  mfem_find_component("${Prefix}" "${DirVar}" "${IncSuffixes}" "${Header}"
    "${LibSuffixes}" "${Lib}" "${IncDoc}" "${LibDoc}")

  if (((NOT Lib) OR ${Prefix}_LIBRARIES) AND
      ((NOT Header) OR ${Prefix}_INCLUDE_DIRS))
    set(Found TRUE)
  else()
    set(Found FALSE)
  endif()

  set(ReqVars "")

  # Check for optional "ADD_COMPONENT" arguments.
  set(I 9) # 9 is the number of required arguments
  while(I LESS ARGC)
    if ("${ARGV${I}}" STREQUAL "CHECK_BUILD")
      # "CHECK_BUILD" has 3 arguments, handled below
      math(EXPR I "${I}+3")
    elseif ("${ARGV${I}}" STREQUAL "ADD_COMPONENT")
      # "ADD_COMPONENT" has 5 arguments:
      # CompPrefix CompIncSuffixes CompHeader CompLibSuffixes CompLib
      math(EXPR I "${I}+1")
      set(CompPrefix "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompIncSuffixes "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompHeader "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompLibSuffixes "${ARGV${I}}")
      math(EXPR I "${I}+1")
      set(CompLib "${ARGV${I}}")
      # Determine if the component is requested.
      list(FIND ${Name}_FIND_COMPONENTS ${CompPrefix} CompIdx)
      if (CompIdx GREATER -1)
        set(CompRequested TRUE)
      else()
        set(CompRequested FALSE)
      endif()
      # Determine if the component is optional or required.
      set(CompRequired ${${Name}_FIND_REQUIRED_${CompPrefix}})
      if (CompRequested)
        set(FullPrefix "${Prefix}_${CompPrefix}")
        mfem_find_component("${FullPrefix}" "${DirVar}"
          "${CompIncSuffixes}" "${CompHeader}"
          "${CompLibSuffixes}" "${CompLib}" "" "")
        if (CompRequired)
          if (CompLib)
            list(APPEND ReqVars ${FullPrefix}_LIBRARIES)
          endif()
          if (CompHeader)
            list(APPEND ReqVars ${FullPrefix}_INCLUDE_DIRS)
          endif()
        endif(CompRequired)
        if (((NOT CompLib) OR ${FullPrefix}_LIBRARIES) AND
            ((NOT CompHeader) OR ${FullPrefix}_INCLUDE_DIRS))
          # Component found
          set(${FullPrefix}_FOUND TRUE CACHE BOOL
              "${Name}/${CompPrefix} was found." FORCE)
          list(APPEND ${Prefix}_LIBRARIES ${${FullPrefix}_LIBRARIES})
          list(APPEND ${Prefix}_INCLUDE_DIRS ${${FullPrefix}_INCLUDE_DIRS})
          if (NOT ${Name}_FIND_QUIETLY)
            # message(STATUS "${Name}: ${CompPrefix}: found")
            message(STATUS
              "${Name}: ${CompPrefix}: ${${FullPrefix}_LIBRARIES}")
            # message(STATUS
            #   "${Name}: ${CompPrefix}: ${${FullPrefix}_INCLUDE_DIRS}")
          endif()
        else()
          # Let FindPackageHandleStandardArgs() handle errors
          if (NOT ${Name}_FIND_QUIETLY)
            message(STATUS "${Name}: ${CompPrefix}: *** NOT FOUND ***")
          endif()
        endif()
      endif(CompRequested)
    else()
      message(FATAL_ERROR "Unknown argument: ${ARGV${I}}")
    endif()
    math(EXPR I "${I}+1")
  endwhile()

  # Add required / optional / alternative packages.
  set(Required "REQUIRED")
  set(Quiet "")
  if (${Name}_FIND_QUIETLY)
    set(Quiet "QUIET")
  endif()
  set(Alternative FALSE)
  foreach(ReqPack IN LISTS ${Name}_REQUIRED_PACKAGES)
    # Parse the pattern: <PackName>[/<CompName>]...
    string(REPLACE "/" ";" PackComps "${ReqPack}")
    list(GET PackComps 0 PackName)
    list(REMOVE_AT PackComps 0)
    set(ReqPack "${PackName}")
    set(ReqPackM "${ReqPack}")
    if (NOT ("${PackComps}" STREQUAL ""))
       set(ReqPackM "${ReqPackM}, COMPONENTS: ${PackComps}")
    endif()
    if (Quiet)
      set(ReqPackM "${ReqPackM} (quiet)")
    endif()
    if ("${ReqPack}" STREQUAL "REQUIRED:")
      set(Required "REQUIRED")
    elseif ("${ReqPack}" STREQUAL "OPTIONAL:")
      set(Required "")
    elseif ("${ReqPack}" STREQUAL "QUIET:")
      set(Quiet "QUIET")
    elseif ("${ReqPack}" STREQUAL "VERBOSE:")
      set(Quiet "")
      if (${Name}_FIND_QUIETLY)
        set(Quiet "QUIET")
      endif()
    elseif ("${ReqPack}" STREQUAL "ALT:")
      set(Alternative TRUE)
    elseif ((NOT Found) AND Alternative)
      set(Alternative FALSE)
      if (NOT ${Name}_FIND_QUIETLY)
        message(STATUS "${Name}: trying alternative package: ${ReqPackM}")
      endif()
      find_package(${ReqPack} ${Quiet} COMPONENTS ${PackComps})
      string(TOUPPER ${ReqPack} ReqPACK)
      if (${ReqPack}_FOUND)
        set(Found TRUE)
        set(${Prefix}_LIBRARIES ${${ReqPack}_LIBRARIES})
        set(${Prefix}_INCLUDE_DIRS ${${ReqPack}_INCLUDE_DIRS})
      elseif (${ReqPACK}_FOUND)
        set(Found TRUE)
        set(${Prefix}_LIBRARIES ${${ReqPACK}_LIBRARIES})
        set(${Prefix}_INCLUDE_DIRS ${${ReqPACK}_INCLUDE_DIRS})
      endif()
    elseif (Alternative)
      set(Alternative FALSE)
    else()
      if (NOT ${Name}_FIND_QUIETLY)
        if (Required)
          message(STATUS "${Name}: looking for required package: ${ReqPackM}")
        else()
          message(STATUS "${Name}: looking for optional package: ${ReqPackM}")
        endif()
      endif()
      string(TOUPPER ${ReqPack} ReqPACK)
      if (NOT (${ReqPack}_FOUND OR ${ReqPACK}_FOUND))
        find_package(${ReqPack} ${Required} ${Quiet} COMPONENTS ${PackComps})
      endif()
      if ("${ReqPack}" STREQUAL "MPI")
        list(APPEND ${Prefix}_LIBRARIES ${MPI_CXX_LIBRARIES})
        list(APPEND ${Prefix}_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
      else()
        if (${ReqPack}_FOUND)
          list(APPEND ${Prefix}_LIBRARIES ${${ReqPack}_LIBRARIES})
          list(APPEND ${Prefix}_INCLUDE_DIRS ${${ReqPack}_INCLUDE_DIRS})
        elseif (${ReqPACK}_FOUND)
          list(APPEND ${Prefix}_LIBRARIES ${${ReqPACK}_LIBRARIES})
          list(APPEND ${Prefix}_INCLUDE_DIRS ${${ReqPACK}_INCLUDE_DIRS})
        endif()
      endif()
    endif()
  endforeach()

  if (NOT ("${${Prefix}_INCLUDE_DIRS}" STREQUAL ""))
    list(INSERT ReqVars 0 ${Prefix}_INCLUDE_DIRS)
    set(ReqHeaders 1)
  endif()
  if (NOT ("${${Prefix}_LIBRARIES}" STREQUAL ""))
    list(INSERT ReqVars 0 ${Prefix}_LIBRARIES)
    set(ReqLibs 1)
  endif()

  if (Found)
    if (ReqLibs)
      list(REMOVE_DUPLICATES ${Prefix}_LIBRARIES)
    endif()
    if (ReqHeaders)
      list(REMOVE_DUPLICATES ${Prefix}_INCLUDE_DIRS)
    endif()
    # Write the updated values to the cache.
    set(${Prefix}_LIBRARIES ${${Prefix}_LIBRARIES} CACHE STRING
        "${LibDoc}" FORCE)
    set(${Prefix}_INCLUDE_DIRS ${${Prefix}_INCLUDE_DIRS} CACHE STRING
        "${IncDoc}" FORCE)
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)

    # Check for optional "CHECK_BUILD" arguments.
    set(I 9) # 9 is the number of required arguments
    while(I LESS ARGC)
      if ("${ARGV${I}}" STREQUAL "CHECK_BUILD")
        math(EXPR I "${I}+1")
        set(TestVar "${ARGV${I}}")
        math(EXPR I "${I}+1")
        set(TestReq "${ARGV${I}}")
        math(EXPR I "${I}+1")
        set(TestSrc "${ARGV${I}}")
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_INCLUDES ${${Prefix}_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${${Prefix}_LIBRARIES})
        set(CMAKE_REQUIRED_QUIET ${${Name}_FIND_QUIETLY})
        check_cxx_source_compiles("${TestSrc}" ${TestVar})
        if (TestReq)
          list(APPEND ReqVars ${TestVar})
        endif()
      elseif("${ARGV${I}}" STREQUAL "ADD_COMPONENT")
        # "ADD_COMPONENT" has 5 arguments, handled above
        math(EXPR I "${I}+5")
      else()
        message(FATAL_ERROR "Unknown argument: ${ARGV${I}}")
      endif()
      math(EXPR I "${I}+1")
    endwhile()
  else()
    set(${Prefix}_FOUND FALSE CACHE BOOL "${Name} was not found." FORCE)
  endif()
  if ("_x_${ReqVars}" STREQUAL "_x_")
    set(ReqVars ${Prefix}_FOUND)
  endif()
  # foreach(ReqVar ${ReqVars})
  #   message(STATUS "${ReqVar}=${${ReqVar}}")
  # endforeach()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(${Name}
    " *** ${Name} not found. Please set ${DirVar}." ${ReqVars})

  if (Found AND ReqLibs AND ReqHeaders AND (NOT ${Name}_FIND_QUIETLY))
    message(STATUS "${Prefix}_INCLUDE_DIRS=${${Prefix}_INCLUDE_DIRS}")
  endif()

endfunction(mfem_find_package)


#
#   Function checking if the code snippet CheckSrc compiles and links using the
#   C++ compiler default include and link paths, searching for a particular
#   library, or no library at all, to link with.
#
#   Checks if the code snippet CheckSrc works with:
#      a) one of the libraries listed in ${Prefix}_LIBRARIES, if any,
#      b) without any library, just standard C/C++, or
#      c) with one of the libraries listed in Lib, if any.
#
#   Defines the variables:
#      ${Prefix}_FOUND
#      ${Prefix}_LIBRARIES (empty if no library is needed)
#
#   If ${Name}_SKIP_STANDARD is true, then check (b) above is skipped.
#
#   If ${Name}_SKIP_FPHSA is true and the package was not found, skip the call
#   to the function find_package_handle_standard_args(...).
#
#   This function is intended to be called from the script Find${Name}.cmake
#
function(mfem_find_library Name Prefix Lib LibDoc CheckVar CheckSrc)

  # Quick return
  if (${Prefix}_FOUND)
    return()
  endif()

  if (NOT ${Name}_FIND_QUIETLY)
    message(STATUS "Looking for ${Name} ...")
  endif()

  include(CheckCXXSourceCompiles)
  foreach(CMAKE_REQUIRED_LIBRARIES ${${Prefix}_LIBRARIES} "" ${Lib})
    unset(${CheckVar} CACHE)
    if (CMAKE_REQUIRED_LIBRARIES OR (NOT ${Name}_SKIP_STANDARD))
      if (NOT ${Name}_FIND_QUIETLY)
        if (CMAKE_REQUIRED_LIBRARIES)
          message(STATUS "   checking library: ${CMAKE_REQUIRED_LIBRARIES}")
        else()
          message(STATUS "   checking library: <standard c/c++>")
        endif()
      endif()
      if ("${CMAKE_REQUIRED_LIBRARIES}" STREQUAL "")
        set(ReqVars ${Prefix}_FOUND)
      else()
        set(ReqVars ${Prefix}_LIBRARIES)
      endif()
      #   CMAKE_REQUIRED_FLAGS = string of compile command line flags
      #   CMAKE_REQUIRED_DEFINITIONS = list of macros to define (-DFOO=bar)
      #   CMAKE_REQUIRED_INCLUDES = list of include directories
      #   CMAKE_REQUIRED_LIBRARIES = list of libraries to link
      #   CMAKE_REQUIRED_QUIET = execute quietly without messages
      set(CMAKE_REQUIRED_QUIET ${Name}_FIND_QUIETLY)
      check_cxx_source_compiles("${CheckSrc}" ${CheckVar})
      if (${CheckVar})
        set(${Prefix}_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} CACHE STRING
            "${LibDoc}" FORCE)
        break()
      endif()
    endif()
  endforeach()

  if (${CheckVar})
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)
  else()
    set(${Prefix}_FOUND FALSE CACHE BOOL "${Name} was not found." FORCE)
  endif()

  if (${Prefix}_FOUND OR (NOT ${Name}_SKIP_FPHSA))
    # Handle REQUIRED etc
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(${Name}
      " *** ${Name} not found." ${ReqVars})
  endif()

endfunction(mfem_find_library)
