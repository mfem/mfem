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

# Defines the following variables:
#   - GLVIS_FOUND
#   - GLVIS_LIBRARIES
#   - GLVIS_INCLUDE_DIRS

if (MFEM_FETCH_GLVIS OR MFEM_FETCH_TPLS)
    message(STATUS "[🔵 GLVis 🔵] Fetch/ExternalProject")

    get_directory_property(COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
    get_directory_property(COMPILE_OPTS COMPILE_OPTIONS)
    get_directory_property(LINK_OPTS LINK_OPTIONS)

    string(REPLACE ";" " " COMPILE_OPTS_STR "${COMPILE_OPTS}")
    string(REPLACE "\"" "\\\"" COMPILE_DEFS_QUOTED_STR "${COMPILE_DEFINITIONS}")
    string(REPLACE ";" " -D" COMPILE_DEFS_STR "-D${COMPILE_DEFS_QUOTED_STR}")
    string(JOIN " " COMPILE_CXX_FLAGS ${COMPILE_OPTS_STR} ${COMPILE_DEFS_STR})

    add_library(GLVIS STATIC IMPORTED)

    include(ExternalProject)

    set(FETCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/fetch)
    set(FETCH_GLVIS "${FETCH_DIR}/glvis")

    ExternalProject_Add(glvis
        GIT_REPOSITORY https://github.com/GLVis/glvis.git
        GIT_TAG stream_sessions
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        PREFIX ${FETCH_GLVIS}
        SOURCE_DIR ${FETCH_GLVIS}/src
        STAMP_DIR ${FETCH_GLVIS}/stamp
        BINARY_DIR ${FETCH_GLVIS}/build
        DEPENDS mfem
        CMAKE_ARGS
            -DCMAKE_VERBOSE_MAKEFILE=ON
            -DMFEM_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_FLAGS:STRING=${COMPILE_CXX_FLAGS}
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build ${FETCH_GLVIS}/build
                            --config $<CONFIG>
                            --target libglvis.a glvis_logo
        BUILD_BYPRODUCTS 
            ${FETCH_GLVIS}/build/lib/libglvis.a
            ${FETCH_GLVIS}/build/share/libglvis_logo.a
        INSTALL_COMMAND "")

    set_target_properties(GLVIS PROPERTIES
        IMPORTED_LOCATION ${FETCH_GLVIS}/build/lib/libglvis.a)

    find_package(OpenGL REQUIRED)
    find_package(GLEW   REQUIRED)
    find_package(SDL2   REQUIRED)
    find_package(PNG    REQUIRED)
    find_package(Freetype REQUIRED)
    find_package(Fontconfig REQUIRED)
    if(APPLE)
        find_library(COCOA_LIBRARY Cocoa)
    endif()
    target_link_libraries(GLVIS INTERFACE
        ${FETCH_GLVIS}/build/share/libglvis_logo.a
        OpenGL::GL
        GLEW::GLEW
        SDL2::SDL2
        PNG::PNG
        Freetype::Freetype
        Fontconfig::Fontconfig)
    if(APPLE)
        target_link_libraries(GLVIS INTERFACE ${COCOA_LIBRARY})
    endif()

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

        find_package(OpenGL REQUIRED)
        list(APPEND GLVIS_LIBRARIES OpenGL::GL)

        find_package(GLEW REQUIRED)
        list(APPEND GLVIS_LIBRARIES GLEW::GLEW)
        
        find_package(SDL2 REQUIRED)
        list(APPEND GLVIS_LIBRARIES SDL2::SDL2)

        find_package(PNG REQUIRED)
        list(APPEND GLVIS_LIBRARIES PNG::PNG)

        find_package(Freetype REQUIRED)
        list(APPEND GLVIS_LIBRARIES Freetype::Freetype)

        find_package(Fontconfig REQUIRED)
        list(APPEND GLVIS_LIBRARIES Fontconfig::Fontconfig)

        find_library(COCOA_LIBRARY Cocoa)
        list(APPEND GLVIS_LIBRARIES ${COCOA_LIBRARY})
endif()
        
message(STATUS "GLVIS_INCLUDE_DIRS: ${GLVIS_INCLUDE_DIRS}")
message(STATUS "GLVIS_LIBRARIES: ${GLVIS_LIBRARIES}")
