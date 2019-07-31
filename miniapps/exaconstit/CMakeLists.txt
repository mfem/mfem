#
# see README.md for copyright and license information
# 

cmake_minimum_required(VERSION 3.1)
if("${CMAKE_VERSION}" VERSION_GREATER 3.3)
    cmake_policy(SET CMP0057 NEW)
endif()

message(STATUS "Using cmake version ${CMAKE_VERSION}")

project(EXACONSTIT LANGUAGES CXX)

if (ENABLE_FORTRAN)
   enable_language(Fortran)
endif()

enable_language(C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

################################
# BLT
################################
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    # We currently don't support this option
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT submodule is not present. "
            "Run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or you can point to an outside of source BLT repository by "
            "defining the variable BLT_SOURCE_DIR on your command line " )
    endif()
endif()

set(ENABLE_GTEST OFF CACHE BOOL "")
set(ENABLE_FRUIT OFF CACHE BOOL "")
# if(ENABLE_TESTS)
#     set(ENABLE_GTEST ON CACHE BOOL "" FORCE)
# endif(ENABLE_TESTS)

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

# set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")
# set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

################################
# Include standard build system logic and options
################################
include(cmake/CMakeBasics.cmake)

################################
# Add source subdirectories
################################
add_subdirectory(src/)

# if(ENABLE_TESTS)
#     add_subdirectory(test)
# endif(ENABLE_TESTS)
