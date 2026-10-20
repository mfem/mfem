# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Defines the following variables:
#   - GLVIS_FOUND
#   - GLVIS_LIBRARIES
#   - GLVIS_INCLUDE_DIRS

# Prefer an imported target. Otherwise use the first variable that names a
# library. Older Find modules set only the variables.
macro(mfem_glvis_dep OUT)
    set(${OUT} "")
    foreach(item ${ARGN})
        if(TARGET "${item}")
            set(${OUT} "${item}")
            break()
        endif()
        if(DEFINED ${item} AND NOT "${${item}}" STREQUAL ""
           AND NOT "${${item}}" MATCHES "NOTFOUND")
            set(${OUT} "${${item}}")
            break()
        endif()
    endforeach()
    if("${${OUT}}" STREQUAL "")
        message(FATAL_ERROR "GLVis dependency not found. Tried: ${ARGN}")
    endif()
endmacro()

function(mfem_glvis_find_deps OUT)
    find_package(OpenGL QUIET)
    find_package(GLEW QUIET)
    find_package(SDL2 QUIET)
    find_package(PNG QUIET)
    find_package(Freetype QUIET)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.14)
        find_package(Fontconfig QUIET)
    else()
        find_library(FONTCONFIG_LIBRARIES NAMES fontconfig)
    endif()

    mfem_glvis_dep(opengl OpenGL::GL OPENGL_LIBRARIES OpenGL_LIBRARIES)
    mfem_glvis_dep(glew GLEW::GLEW GLEW_LIBRARIES)
    mfem_glvis_dep(sdl SDL2::SDL2 SDL2_LIBRARIES SDL2_LIBRARY)
    mfem_glvis_dep(png PNG::PNG PNG_LIBRARIES)
    mfem_glvis_dep(freetype Freetype::Freetype FREETYPE_LIBRARIES)
    mfem_glvis_dep(fontconfig Fontconfig::Fontconfig FONTCONFIG_LIBRARIES
        Fontconfig_LIBRARIES)

    set(deps)
    list(APPEND deps ${opengl} ${glew} ${sdl} ${png} ${freetype} ${fontconfig})
    if(APPLE)
        # REQUIRED on find_library needs CMake 3.18. MFEM requires 3.12.
        find_library(COCOA_LIBRARY Cocoa)
        if(NOT COCOA_LIBRARY)
            message(FATAL_ERROR "Cocoa library not found.")
        endif()
        list(APPEND deps "${COCOA_LIBRARY}")
    endif()
    set(${OUT} "${deps}" PARENT_SCOPE)
endfunction()

# Apple SDK headers included by glvis's Objective-C++ are not system
# headers under Homebrew Clang. Leave -Werror on MFEM.
macro(mfem_glvis_without_werror OUT IN)
    if(APPLE)
        string(REGEX REPLACE "(^| )-Werror(=[^ ]*)?" " " ${OUT} "${IN}")
        string(REGEX REPLACE "(^| )-pedantic-errors($| )" " "
               ${OUT} "${${OUT}}")
        string(STRIP "${${OUT}}" ${OUT})
    else()
        set(${OUT} "${IN}")
    endif()
endmacro()

set(_glvis_lib_leaf
    "${CMAKE_STATIC_LIBRARY_PREFIX}glvis${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(_glvis_logo_leaf
    "${CMAKE_STATIC_LIBRARY_PREFIX}glvis_logo${CMAKE_STATIC_LIBRARY_SUFFIX}")

if (MFEM_FETCH_GLVIS OR MFEM_FETCH_TPLS)
    message(STATUS "GLVis: Fetch/ExternalProject")

    # COMPILE_OPTIONS keeps the CMake-only "SHELL:" prefix until the generator
    # expands it. CMAKE_CXX_FLAGS honors that prefix only at the start of the
    # string, so drop it here and forward the remaining compiler flags.
    # Forced includes (the NVTX headers) also need the directory compile
    # definitions and the MFEM tree: glvis's compiler check sees only
    # CMAKE_CXX_FLAGS. -isystem keeps those paths behind glvis's own -I paths.
    get_directory_property(COMPILE_OPTS COMPILE_OPTIONS)
    get_directory_property(_glvis_compile_defs COMPILE_DEFINITIONS)
    set(_glvis_cxx_flags "")
    set(_glvis_needs_mfem_includes FALSE)
    foreach(opt IN LISTS COMPILE_OPTS)
        if(opt MATCHES "^SHELL:")
            string(SUBSTRING "${opt}" 6 -1 opt)
        endif()
        if(opt MATCHES "(^| )-include( |$)")
            set(_glvis_needs_mfem_includes TRUE)
        endif()
        if(APPLE AND (opt MATCHES "^-Werror(=.+)?$"
                      OR opt STREQUAL "-pedantic-errors"))
            continue()
        endif()
        list(APPEND _glvis_cxx_flags "${opt}")
    endforeach()
    foreach(def IN LISTS _glvis_compile_defs)
        # CMAKE_CXX_FLAGS is parsed like a shell command, which strips quotes.
        # Keep them escaped so -DNAME="value" still reaches the compiler.
        string(REPLACE "\"" "\\\"" def "${def}")
        list(APPEND _glvis_cxx_flags "-D${def}")
    endforeach()
    if(_glvis_needs_mfem_includes)
        list(APPEND _glvis_cxx_flags
            "-isystem ${PROJECT_SOURCE_DIR}"
            "-isystem ${PROJECT_BINARY_DIR}")
    endif()
    # Homebrew Clang searches an -isystem of the macOS SDK usr/include
    # (FindZLIB inside GLVis) before libc++'s own C headers. -I is
    # searched first, so <stdlib.h> stays the libc++ wrapper.
    if(APPLE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
            OUTPUT_VARIABLE _glvis_resource
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _glvis_resource_rc)
        if(_glvis_resource_rc EQUAL 0 AND _glvis_resource)
            get_filename_component(_glvis_llvm_root
                "${_glvis_resource}" DIRECTORY)
            get_filename_component(_glvis_llvm_root
                "${_glvis_llvm_root}" DIRECTORY)
            get_filename_component(_glvis_llvm_root
                "${_glvis_llvm_root}" DIRECTORY)
            set(_glvis_libcxx "${_glvis_llvm_root}/include/c++/v1")
            if(EXISTS "${_glvis_libcxx}/stdlib.h")
                list(APPEND _glvis_cxx_flags "-I${_glvis_libcxx}")
            endif()
        endif()
    endif()

    set(_glvis_extra_cmake_args "")
    if(CMAKE_OSX_SYSROOT)
        list(APPEND _glvis_extra_cmake_args
            "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
    endif()
    # Directory COMPILE_OPTIONS are not CMAKE_CXX_FLAGS. Replacing the
    # parent flags with only those options drops an ABI -D and the
    # per-config flags, such as -pedantic on CMAKE_CXX_FLAGS_DEBUG.
    list(JOIN _glvis_cxx_flags " " _glvis_dir_flags)
    mfem_glvis_without_werror(_glvis_parent "${CMAKE_CXX_FLAGS}")
    string(STRIP "${_glvis_parent} ${_glvis_dir_flags}" COMPILE_CXX_FLAGS)
    foreach(_glvis_bt DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        mfem_glvis_without_werror(_glvis_bt_flags
            "${CMAKE_CXX_FLAGS_${_glvis_bt}}")
        if(NOT "${_glvis_bt_flags}" STREQUAL "")
            list(APPEND _glvis_extra_cmake_args
                "-DCMAKE_CXX_FLAGS_${_glvis_bt}:STRING=${_glvis_bt_flags}")
        endif()
    endforeach()

    add_library(GLVIS STATIC IMPORTED)

    include(ExternalProject)

    set(FETCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/fetch)
    set(FETCH_GLVIS "${FETCH_DIR}/glvis")

    # Single-config archives live in the target's binary directory.
    # Multi-config generators add the configuration name. $<CONFIG> in
    # BUILD_BYPRODUCTS needs CMake 3.20.
    if(CMAKE_CONFIGURATION_TYPES)
        if(CMAKE_VERSION VERSION_LESS 3.20)
            set(_glvis_cfg "${CMAKE_CFG_INTDIR}/")
        else()
            set(_glvis_cfg "$<CONFIG>/")
        endif()
    else()
        set(_glvis_cfg "")
    endif()
    set(GLVIS_LIBRARY
        "${FETCH_GLVIS}/build/lib/${_glvis_cfg}${_glvis_lib_leaf}")
    set(GLVIS_LOGO_LIBRARY
        "${FETCH_GLVIS}/build/share/${_glvis_cfg}${_glvis_logo_leaf}")

    ExternalProject_Add(glvis
        GIT_REPOSITORY https://github.com/GLVis/glvis.git
        # stream_sessions at 83aa1356 (2026-06-30). A branch name moves,
        # so the same MFEM commit would fetch a different GLVis.
        GIT_TAG 83aa1356392ad6168f765364f0e70b616dbb88ae
        # A commit hash is not a branch; --depth 1 cannot clone it.
        GIT_SHALLOW FALSE
        UPDATE_DISCONNECTED TRUE
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        PREFIX ${FETCH_GLVIS}
        SOURCE_DIR ${FETCH_GLVIS}/src
        STAMP_DIR ${FETCH_GLVIS}/stamp
        BINARY_DIR ${FETCH_GLVIS}/build
        DEPENDS mfem
        CMAKE_ARGS
            # -DCMAKE_VERBOSE_MAKEFILE=ON
            -DMFEM_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_FLAGS:STRING=${COMPILE_CXX_FLAGS}
            ${_glvis_extra_cmake_args}
            -DGLVIS_BUILD_LIB_ONLY=ON
        BUILD_COMMAND ${CMAKE_COMMAND}
            --build ${FETCH_GLVIS}/build
            --config $<CONFIG>
            --target glvis glvis_logo
        BUILD_BYPRODUCTS
            ${GLVIS_LIBRARY}
            ${GLVIS_LOGO_LIBRARY}
        INSTALL_COMMAND "")

    # Targets that link GLVIS, and do not depend on exec_prerequisites,
    # still need a rule that builds the archive.
    add_dependencies(GLVIS glvis)

    set_target_properties(GLVIS PROPERTIES
        IMPORTED_LOCATION ${GLVIS_LIBRARY})

    mfem_glvis_find_deps(GLVIS_DEP_LIBS)
    target_link_libraries(GLVIS INTERFACE
        ${GLVIS_LOGO_LIBRARY}
        ${GLVIS_DEP_LIBS})

    set(GLVIS_LIBRARIES
        ${GLVIS_LIBRARY}
        ${GLVIS_LOGO_LIBRARY}
        ${GLVIS_DEP_LIBS})
    set(GLVIS_FOUND TRUE)
    return()
endif()

message(STATUS "[🔵 GLVis 🔵] Find pre-installed package")

include(MfemCmakeUtilities)
mfem_find_package(GLVis GLVIS GLVIS_DIR
        "include" "lib/glwindow.hpp"
        "lib" "build/lib/${_glvis_lib_leaf}"
        "Paths to headers required by GLVis"
        "Libraries required by GLVis")

if (GLVIS_FOUND)
        set(GLVIS_INCLUDE_DIRS ${GLVIS_INCLUDE_DIRS}/lib)

        # REQUIRED on find_library needs CMake 3.18. MFEM requires 3.12.
        find_library(GLVIS_LOGO_LIBRARY
                NAMES glvis_logo
                PATHS ${GLVIS_DIR}/build/share
                NO_DEFAULT_PATH)
        if(NOT GLVIS_LOGO_LIBRARY)
            message(FATAL_ERROR
                "GLVis logo library not found in ${GLVIS_DIR}/build/share")
        endif()
        list(APPEND GLVIS_LIBRARIES ${GLVIS_LOGO_LIBRARY})

        mfem_glvis_find_deps(GLVIS_DEP_LIBS)
        list(APPEND GLVIS_LIBRARIES ${GLVIS_DEP_LIBS})
endif()

message(STATUS "GLVIS_INCLUDE_DIRS: ${GLVIS_INCLUDE_DIRS}")
message(STATUS "GLVIS_LIBRARIES: ${GLVIS_LIBRARIES}")
