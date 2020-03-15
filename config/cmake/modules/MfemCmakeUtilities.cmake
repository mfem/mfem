# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

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
    # If CUDA is enabled, tag source files to be compiled with nvcc.
    if (MFEM_USE_CUDA)
      set_property(SOURCE ${SRC_FILE} PROPERTY LANGUAGE CUDA)
    endif()

    get_filename_component(SRC_FILENAME ${SRC_FILE} NAME)

    string(REPLACE ".cpp" "" EXE_NAME "${EXE_PREFIX}${SRC_FILENAME}")
    add_executable(${EXE_NAME} ${SRC_FILE})
    add_dependencies(${MFEM_ALL_EXAMPLES_TARGET_NAME} ${EXE_NAME})
    if (EXE_NEEDED_BY)
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

  # If CUDA is enabled, tag source files to be compiled with nvcc.
  if (MFEM_USE_CUDA)
    set_property(SOURCE ${MAIN_LIST} ${EXTRA_SOURCES_LIST}
      PROPERTY LANGUAGE CUDA)
    list(TRANSFORM EXTRA_OPTIONS_LIST PREPEND "-Xcompiler=")
  endif()

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
      find_library(${Prefix}_LIBRARY ${Lib}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${LibSuffixes}
        NO_DEFAULT_PATH
        DOC "${LibDoc}")
    endif()
    find_library(${Prefix}_LIBRARY ${Lib}
      PATH_SUFFIXES ${LibSuffixes}
      DOC "${LibDoc}")
  endif()

  if (Header)
    if (${DirVar} OR EnvDirVar)
      find_path(${Prefix}_INCLUDE_DIR ${Header}
        HINTS ${${DirVar}} ENV ${DirVar}
        PATH_SUFFIXES ${IncSuffixes}
        NO_DEFAULT_PATH
        DOC "${IncDoc}")
    endif()
    find_path(${Prefix}_INCLUDE_DIR ${Header}
      PATH_SUFFIXES ${IncSuffixes}
      DOC "${IncDoc}")
  endif()

endfunction(mfem_find_component)


#   MFEM version of find_package that searches for header/library and if
#   successful, optionally checks building (compile + link) one or more given
#   code snippets. Additionally, a list of required/optional/alternative
#   packages (given by ${Name}_REQUIRED_PACKAGES) are searched for and added to
#   the ${Prefix}_INCLUDE_DIRS and ${Prefix}_LIBRARIES lists. The variable
#   ${Name}_REQUIRED_LIBRARIES can be set to spcecify any additional libraries
#   that are needed. This function defines the following CACHE variables:
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

  # If we have the TPL_ versions of _INCLUDE_DIRS and _LIBRARIES then set the
  # standard ${Prefix} versions
  if (TPL_${Prefix}_INCLUDE_DIRS)
    set(${Prefix}_INCLUDE_DIRS ${TPL_${Prefix}_INCLUDE_DIRS} CACHE STRING
      "TPL_${Prefix}_INCLUDE_DIRS was found." FORCE)
  endif()
  if (TPL_${Prefix}_LIBRARIES)
    set(${Prefix}_LIBRARIES ${TPL_${Prefix}_LIBRARIES} CACHE STRING
      "TPL_${Prefix}_LIBRARIES was found." FORCE)
  endif()

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

  if (((NOT Lib) OR ${Prefix}_LIBRARY) AND
      ((NOT Header) OR ${Prefix}_INCLUDE_DIR))
    set(Found TRUE)
  else()
    set(Found FALSE)
  endif()
  set(${Prefix}_LIBRARIES ${${Prefix}_LIBRARY})
  set(${Prefix}_INCLUDE_DIRS ${${Prefix}_INCLUDE_DIR})

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
            list(APPEND ReqVars ${FullPrefix}_LIBRARY)
          endif()
          if (CompHeader)
            list(APPEND ReqVars ${FullPrefix}_INCLUDE_DIR)
          endif()
        endif(CompRequired)
        if (((NOT CompLib) OR ${FullPrefix}_LIBRARY) AND
            ((NOT CompHeader) OR ${FullPrefix}_INCLUDE_DIR))
          # Component found
          list(APPEND ${Prefix}_LIBRARIES ${${FullPrefix}_LIBRARY})
          list(APPEND ${Prefix}_INCLUDE_DIRS ${${FullPrefix}_INCLUDE_DIR})
          if (NOT ${Name}_FIND_QUIETLY)
            message(STATUS
              "${Name}: ${CompPrefix}: ${${FullPrefix}_LIBRARY}")
            # message(STATUS
            #   "${Name}: ${CompPrefix}: ${${FullPrefix}_INCLUDE_DIR}")
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
      # Do not add ${Required} here, since that will prevent other potential
      # alternative packages from being found.
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
    elseif (Found)
      if (NOT ${Name}_FIND_QUIETLY)
        if (Required)
          message(STATUS "${Name}: looking for required package: ${ReqPackM}")
        else()
          message(STATUS "${Name}: looking for optional package: ${ReqPackM}")
        endif()
      endif()
      string(TOUPPER ${ReqPack} ReqPACK)
      if (NOT (${ReqPack}_FOUND OR ${ReqPACK}_FOUND))
        if (NOT ${ReqPack}_TARGET_NAMES)
          find_package(${ReqPack} ${Required} ${Quiet} COMPONENTS ${PackComps})
        else()
          foreach(_target ${ReqPack} ${${ReqPack}_TARGET_NAMES})
            # Do not use ${Required} here:
            find_package(${_target} NAMES ${_target} ${ReqPack} ${Quiet}
              COMPONENTS ${PackComps})
            string(TOUPPER ${_target} _TARGET)
            if (${_target}_FOUND OR ${_TARGET}_FOUND)
              set(${ReqPack}_FOUND TRUE)
              break()
            endif()
          endforeach()
          if (${Required} AND NOT ${ReqPack}_FOUND)
            message(FATAL_ERROR " *** Required package ${ReqPack} not found."
              "Checked target names: ${ReqPack} ${${ReqPack}_TARGET_NAMES}")
          endif()
        endif()
      endif()
      if (Required AND NOT (${ReqPack}_FOUND OR ${ReqPACK}_FOUND))
        message(FATAL_ERROR " --------- INTERNAL ERROR")
      endif()
      if ("${ReqPack}" STREQUAL "MPI" AND MPI_CXX_FOUND)
        list(APPEND ${Prefix}_LIBRARIES ${MPI_CXX_LIBRARIES})
        list(APPEND ${Prefix}_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
      elseif (${ReqPack}_FOUND OR ${ReqPACK}_FOUND)
        if (${ReqPack}_FOUND)
          set(_Pack ${ReqPack})
        else()
          set(_Pack ${ReqPACK})
        endif()
        set(_Pack_LIBS)
        set(_Pack_INCS)
        # - ${_Pack}_CONFIG is defined by find_package() when a config file was
        #   loaded
        # - If ${ReqPack}_TARGET_NAMES is defined, use target mode
        if (NOT ((DEFINED ${_Pack}_CONFIG) OR
                 (DEFINED ${ReqPack}_TARGET_NAMES)))
          # Defined variables expected:
          # - ${ReqPack}_LIB_VARS, optional, default: ${_Pack}_LIBRARIES
          # - ${ReqPack}_INCLUDE_VARS, optional, default: ${_Pack}_INCLUDE_DIRS
          set(_lib_vars ${${ReqPack}_LIB_VARS})
          if (NOT _lib_vars)
            set(_lib_vars ${_Pack}_LIBRARIES)
          endif()
          foreach (_var ${_lib_vars})
            if (${_var})
              list(APPEND _Pack_LIBS ${${_var}})
            endif()
          endforeach()
          # Includes
          set(_inc_vars ${${ReqPack}_INCLUDE_VARS})
          if (NOT _inc_vars)
            set(_inc_vars ${_Pack}_INCLUDE_DIRS)
          endif()
          foreach (_include ${_inc_vars})
            # message(STATUS "${Name}: ${ReqPack}: ${_include}")
            if (${_include})
              list(APPEND _Pack_INCS ${${_include}})
            endif()
          endforeach()
        else()
          # Target mode: check for a valid target:
          # - an entry in the variable ${ReqPack}_TARGET_NAMES (optional)
          # - ${_Pack}
          # Other optional variables:
          # - ${ReqPack}_IMPORT_CONFIG, default value: "RELEASE"
          # - ${ReqPack}_TARGET_FORCE, default value: "FALSE"
          set(TargetName)
          foreach (_target ${${ReqPack}_TARGET_NAMES} ${_Pack})
            if (TARGET ${_target})
              set(TargetName ${_target})
              break()
            endif()
          endforeach()
          if ("${TargetName}" STREQUAL "")
            message(FATAL_ERROR " *** ${ReqPack}: unknown target. "
              "Please set ${ReqPack}_TARGET_NAMES.")
          endif()
          get_target_property(IsImported ${TargetName} IMPORTED)
          if (IsImported)
            set(ImportConfig ${${ReqPack}_IMPORT_CONFIG})
            if (NOT ImportConfig)
              set(ImportConfig RELEASE)
            endif()
            get_target_property(ImpConfigs ${TargetName} IMPORTED_CONFIGURATIONS)
            list(FIND ImpConfigs ${ImportConfig} _Index)
            if (_Index EQUAL -1)
              message(FATAL_ERROR " *** ${ReqPack}: configuration "
                "${ImportConfig} not found. Set ${ReqPack}_IMPORT_CONFIG "
                "from the list: ${ImpConfigs}.")
            endif()
          endif()
          # Set _Pack_LIBS
          if (NOT IsImported OR ${ReqPack}_TARGET_FORCE)
            # Set _Pack_LIBS to be the target itself
            set(_Pack_LIBS ${TargetName})
            if (NOT ${Name}_FIND_QUIETLY)
              message(STATUS "Found ${ReqPack}: ${_Pack_LIBS} (target)")
            endif()
          else()
            # Set _Pack_LIBS from the target properties for ImportConfig
            foreach (_prop IMPORTED_LOCATION_${ImportConfig}
                IMPORTED_LINK_INTERFACE_LIBRARIES_${ImportConfig})
              get_target_property(_value ${TargetName} ${_prop})
              if (_value)
                list(APPEND _Pack_LIBS ${_value})
              endif()
            endforeach()
            if (NOT ${Name}_FIND_QUIETLY)
              message(STATUS
                "Imported ${ReqPack}[${ImportConfig}]: ${_Pack_LIBS}")
            endif()
          endif()
          # Set _Pack_INCS
          foreach (_prop INCLUDE_DIRECTORIES)
            get_target_property(_value ${TargetName} ${_prop})
            if (_value)
              list(APPEND _Pack_INCS ${_value})
            endif()
          endforeach()
        endif()
        # _Pack_LIBS and _Pack_INCS should be fully defined here
        list(APPEND ${Prefix}_LIBRARIES ${_Pack_LIBS})
        list(APPEND ${Prefix}_INCLUDE_DIRS ${_Pack_INCS})
      endif()
    endif()
  endforeach()

  if (Found AND ${Name}_REQUIRED_LIBRARIES)
    list(APPEND ${Prefix}_LIBRARIES ${${Name}_REQUIRED_LIBRARIES})
  endif()

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
          if (NOT ${TestVar})
            set(Found FALSE)
            unset(${TestVar} CACHE)
          endif()
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
  endif()
  if ("_x_${ReqVars}" STREQUAL "_x_")
    set(${Prefix}_FOUND ${Found})
    set(ReqVars ${Prefix}_FOUND)
  endif()
  # foreach(ReqVar ${ReqVars})
  #   message(STATUS " *** ${ReqVar}=${${ReqVar}}")
  #   get_property(IsCached CACHE ${ReqVar} PROPERTY "VALUE" SET)
  #   if (IsCached)
  #     get_property(CachedVal CACHE ${ReqVar} PROPERTY "VALUE")
  #     message(STATUS " *** ${ReqVar}[cached]=${CachedVal}")
  #   endif()
  # endforeach()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(${Name}
    " *** ${Name} not found. Please set ${DirVar}." ${ReqVars})

  string(TOUPPER ${Name} UName)
  if (${UName}_FOUND)
    # Write the ${Prefix}_* variables to the cache.
    set(${Prefix}_LIBRARIES ${${Prefix}_LIBRARIES} CACHE STRING
        "${LibDoc}" FORCE)
    set(${Prefix}_INCLUDE_DIRS ${${Prefix}_INCLUDE_DIRS} CACHE STRING
        "${IncDoc}" FORCE)
    set(${Prefix}_FOUND TRUE CACHE BOOL "${Name} was found." FORCE)
    if (ReqHeaders AND (NOT ${Name}_FIND_QUIETLY))
      message(STATUS "${Prefix}_INCLUDE_DIRS=${${Prefix}_INCLUDE_DIRS}")
    endif()
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


#
#   Function that creates 'config.mk' from 'config.mk.in' for the both the
#   build- and the install-locations and define install rules for 'config.mk'
#   and 'test.mk'.
#
function(mfem_export_mk_files)

  # Define a few auxiliary variables (not written to 'config.mk')
  string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
  # CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG -> '-Wl,-rpath,'
  set(shared_link_flag ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG})
  if (NOT shared_link_flag)
    set(shared_link_flag "-Wl,-rpath,")
  endif()

  # Convert Boolean vars to YES/NO without writting the values to cache
  set(CONFIG_MK_BOOL_VARS MFEM_USE_MPI MFEM_USE_METIS MFEM_USE_METIS_5
      MFEM_DEBUG MFEM_USE_EXCEPTIONS MFEM_USE_ZLIB MFEM_USE_LIBUNWIND
      MFEM_USE_LAPACK MFEM_THREAD_SAFE MFEM_USE_OPENMP MFEM_USE_LEGACY_OPENMP
      MFEM_USE_MEMALLOC MFEM_USE_SUNDIALS MFEM_USE_MESQUITE MFEM_USE_SUITESPARSE
      MFEM_USE_SUPERLU MFEM_USE_STRUMPACK MFEM_USE_GECKO MFEM_USE_GNUTLS
      MFEM_USE_GSLIB MFEM_USE_NETCDF MFEM_USE_PETSC MFEM_USE_MPFR MFEM_USE_SIDRE
      MFEM_USE_CONDUIT MFEM_USE_PUMI MFEM_USE_CUDA MFEM_USE_OCCA MFEM_USE_RAJA
      MFEM_USE_UMPIRE)
  foreach(var ${CONFIG_MK_BOOL_VARS})
    if (${var})
      set(${var} YES)
    else()
      set(${var} NO)
    endif()
  endforeach()
  # TODO: Add support for MFEM_USE_CUDA=YES
  set(MFEM_CXX ${CMAKE_CXX_COMPILER})
  set(MFEM_CPPFLAGS "")
  string(STRIP "${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${CMAKE_CXX_FLAGS}"
         MFEM_CXXFLAGS)
  set(MFEM_TPLFLAGS "")
  foreach(dir ${MFEM_TPL_INCLUDE_DIRS})
    set(MFEM_TPLFLAGS "${MFEM_TPLFLAGS} -I${dir}")
  endforeach()
  # TODO: MFEM_TPLFLAGS: add other TPL flags, in addition to the -I flags.
  set(MFEM_INCFLAGS "-I\$(MFEM_INC_DIR) \$(MFEM_TPLFLAGS)")
  set(MFEM_PICFLAG "")
  if (BUILD_SHARED_LIBS)
    set(MFEM_PICFLAG "${CMAKE_SHARED_LIBRARY_CXX_FLAGS}")
  endif()
  set(MFEM_FLAGS "\$(MFEM_CPPFLAGS) \$(MFEM_CXXFLAGS) \$(MFEM_INCFLAGS)")
  # TPL link flags: set below
  set(MFEM_EXT_LIBS "")
  if (BUILD_SHARED_LIBS)
    set(MFEM_LIBS "${shared_link_flag}\$(MFEM_LIB_DIR) -L\$(MFEM_LIB_DIR)")
    set(MFEM_LIBS "${MFEM_LIBS} -lmfem \$(MFEM_EXT_LIBS)")
    if (APPLE)
      set(SO_VER ".${mfem_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
      set(SO_VER "${CMAKE_SHARED_LIBRARY_SUFFIX}.${mfem_VERSION}")
    endif()
    set(MFEM_LIB_FILE "\$(MFEM_LIB_DIR)/libmfem${SO_VER}")
    set(MFEM_SHARED YES)
    set(MFEM_STATIC NO)
  else()
    set(MFEM_LIBS "-L\$(MFEM_LIB_DIR) -lmfem \$(MFEM_EXT_LIBS)")
    set(MFEM_LIB_FILE "\$(MFEM_LIB_DIR)/libmfem.a")
    set(MFEM_SHARED NO)
    set(MFEM_STATIC YES)
  endif()
  set(MFEM_BUILD_TAG "${CMAKE_SYSTEM}")
  set(MFEM_PREFIX "${CMAKE_INSTALL_PREFIX}")
  # For the next 4 variable, these are the values for the build-tree version of
  # 'config.mk'
  set(MFEM_INC_DIR "${PROJECT_BINARY_DIR}")
  set(MFEM_LIB_DIR "${PROJECT_BINARY_DIR}")
  set(MFEM_TEST_MK "${PROJECT_SOURCE_DIR}/config/test.mk")
  set(MFEM_CONFIG_EXTRA "MFEM_BUILD_DIR ?= ${PROJECT_BINARY_DIR}")
  set(MFEM_MPIEXEC ${MPIEXEC})
  if (NOT MFEM_MPIEXEC)
    set(MFEM_MPIEXEC "mpirun")
  endif()
  set(MFEM_MPIEXEC_NP ${MPIEXEC_NUMPROC_FLAG})
  if (NOT MFEM_MPIEXEC_NP)
    set(MFEM_MPIEXEC_NP "-np")
  endif()
  # MFEM_MPI_NP is already set
  # Define the variable 'MFEM_EXT_LIBS': handle PUMI libs
  if ("${MFEM_USE_PUMI}" STREQUAL "YES")
    message(STATUS "simmodsuite_dir = '${SIMMODSUITE_DIR}'")
    get_target_property(liblist ${PUMI_LIBRARIES} INTERFACE_LINK_LIBRARIES)
    set(pumi_dep_libs "${liblist}")
    foreach(pumilib ${liblist})
      get_target_property(libdeps ${pumilib} INTERFACE_LINK_LIBRARIES)
      if (NOT "${libdeps}" MATCHES "libdeps-NOTFOUND")
        list(APPEND pumi_dep_libs ${libdeps})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES pumi_dep_libs)
    foreach(pumilib ${pumi_dep_libs})
      unset(lib CACHE)
      string(REGEX REPLACE "^SCOREC::" "" libname ${pumilib})
      string(FIND "${pumilib}" ".a" staticlib)
      string(FIND "${pumilib}" ".so" sharedlib)
      find_library(lib ${libname} PATHS ${PUMI_DIR}/lib NO_DEFUALT_PATH)
      if (NOT "${sharedlib}" MATCHES "-1" OR
          NOT "${staticlib}" MATCHES "-1"   )
        set(MFEM_EXT_LIBS "${pumilib} ${MFEM_EXT_LIBS}")
      elseif (NOT "${lib}" MATCHES "lib-NOTFOUND")
        set(MFEM_EXT_LIBS "${lib} ${MFEM_EXT_LIBS}")
      elseif ("${lib}" MATCHES "lib-NOTFOUND" AND
              NOT "${libname}" MATCHES "can" AND
              NOT "${libname}" MATCHES "pthread")
        message(FATAL_ERROR "SCOREC lib ${libname} not found")
      endif()
    endforeach()
  endif()
  # Define the variable 'MFEM_EXT_LIBS': handle other (not PUMI) libs
  foreach(lib ${TPL_LIBRARIES})
    get_filename_component(suffix ${lib} EXT)
    # handle interfaces (e.g., SCOREC::apf)
    if ("${lib}" MATCHES "SCOREC::.*" OR "${lib}" MATCHES "Ginkgo::.*")
    elseif (NOT "${lib}" MATCHES "SCOREC::.*" AND "${lib}" MATCHES ".*::.*")
      message(FATAL_ERROR "***** interface lib found ... exiting *****")
      # handle static and shared libs
    elseif ("${suffix}" STREQUAL "${CMAKE_SHARED_LIBRARY_SUFFIX}")
      get_filename_component(dir ${lib} DIRECTORY)
      get_filename_component(fullLibName ${lib} NAME_WE)
      string(REGEX REPLACE "^lib" "" libname ${fullLibName})
      set(MFEM_EXT_LIBS
          "${MFEM_EXT_LIBS} ${shared_link_flag}${dir} -L${dir} -l${libname}")
    else()
      set(MFEM_EXT_LIBS "${MFEM_EXT_LIBS} ${lib}")
    endif()
  endforeach()

  # Create the build-tree version of 'config.mk'
  configure_file(
    "${PROJECT_SOURCE_DIR}/config/config.mk.in"
    "${PROJECT_BINARY_DIR}/config/config.mk")
  # Copy 'test.mk' from the source-tree to the build-tree
  configure_file(
    "${PROJECT_SOURCE_DIR}/config/test.mk"
    "${PROJECT_BINARY_DIR}/config/test.mk" COPYONLY)

  # Update variables for the install-tree version of 'config.mk'
  set(MFEM_INC_DIR "${CMAKE_INSTALL_PREFIX}/include")
  set(MFEM_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  set(MFEM_TEST_MK "${CMAKE_INSTALL_PREFIX}/share/mfem/test.mk")
  set(MFEM_CONFIG_EXTRA "")

  # Create the install-tree version of 'config.mk'
  configure_file(
    "${PROJECT_SOURCE_DIR}/config/config.mk.in"
    "${PROJECT_BINARY_DIR}/config/config-install.mk")

  # Install rules for 'config.mk' and 'test.mk'
  install(FILES ${PROJECT_SOURCE_DIR}/config/test.mk
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/mfem/)
  install(FILES ${PROJECT_BINARY_DIR}/config/config-install.mk
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/mfem/ RENAME config.mk)

endfunction()
