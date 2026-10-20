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
        find_library(COCOA_LIBRARY Cocoa REQUIRED)
        list(APPEND deps "${COCOA_LIBRARY}")
    endif()
    set(${OUT} "${deps}" PARENT_SCOPE)
endfunction()

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
        # Apple SDK headers included by glvis's Objective-C++ are not system
        # headers under Homebrew Clang's framework path. Leave warning-as-error
        # flags on MFEM and do not apply them to the fetched library.
        if(APPLE AND (opt MATCHES "^-Werror(=.+)?$" OR opt STREQUAL "-pedantic-errors"))
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

    set(_glvis_extra_cmake_args "")
    if(CMAKE_OSX_SYSROOT)
        list(APPEND _glvis_extra_cmake_args
            "-DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}")
    endif()
    list(JOIN _glvis_cxx_flags " " COMPILE_CXX_FLAGS)

    cmake_host_system_information(RESULT NCPU QUERY NUMBER_OF_LOGICAL_CORES)

    add_library(GLVIS STATIC IMPORTED)

    include(ExternalProject)

    set(FETCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/fetch)
    set(FETCH_GLVIS "${FETCH_DIR}/glvis")

    ExternalProject_Add(glvis
        GIT_REPOSITORY https://github.com/GLVis/glvis.git
        GIT_TAG stream_sessions
        GIT_SHALLOW TRUE
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
            --parallel ${NCPU}
        BUILD_BYPRODUCTS
            ${FETCH_GLVIS}/build/lib/libglvis.a
            ${FETCH_GLVIS}/build/share/libglvis_logo.a
        INSTALL_COMMAND "")

    set_target_properties(GLVIS PROPERTIES
        IMPORTED_LOCATION ${FETCH_GLVIS}/build/lib/libglvis.a)

    mfem_glvis_find_deps(GLVIS_DEP_LIBS)
    target_link_libraries(GLVIS INTERFACE
        ${FETCH_GLVIS}/build/share/libglvis_logo.a
        ${GLVIS_DEP_LIBS})

    set(GLVIS_FOUND TRUE)
    return()
endif()

message(STATUS "[🔵 GLVis 🔵] Find pre-installed package")

include(MfemCmakeUtilities)
mfem_find_package(GLVis GLVIS GLVIS_DIR
        "include" "lib/glwindow.hpp"
        "lib" "build/lib/libglvis.a"
        "Paths to headers required by GLVis"
        "Libraries required by GLVis")

if (GLVIS_FOUND)
        set(GLVIS_INCLUDE_DIRS ${GLVIS_INCLUDE_DIRS}/lib)

        find_library(GLVIS_LOGO_LIBRARY
                NAMES glvis_logo
                PATHS ${GLVIS_DIR}/build/share
                NO_DEFAULT_PATH
                REQUIRED)
        list(APPEND GLVIS_LIBRARIES ${GLVIS_LOGO_LIBRARY})

        mfem_glvis_find_deps(GLVIS_DEP_LIBS)
        list(APPEND GLVIS_LIBRARIES ${GLVIS_DEP_LIBS})
endif()

message(STATUS "GLVIS_INCLUDE_DIRS: ${GLVIS_INCLUDE_DIRS}")
message(STATUS "GLVIS_LIBRARIES: ${GLVIS_LIBRARIES}")
