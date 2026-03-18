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
