# if (NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
#       "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel." FORCE)
# endif()

# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     add_compile_options(-fsanitize=address -O0)
#     add_link_options(-fsanitize=address)
# endif()


## MFEM specific settings #####################################################
set(MFEM_DIR ${CMAKE_CURRENT_SOURCE_DIR})

## Linux, Darwin specific settings ############################################
if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    # include_directories(/usr/include)
elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(CXX /opt/homebrew/opt/llvm/bin/clang++)
    include_directories(/opt/homebrew/include)
    find_package(fmt CONFIG REQUIRED)
    add_link_options(-L/opt/homebrew/opt/fmt/lib -lfmt)
else()
   message(FATAL_ERROR "Unsupported system")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose makefiles.")

option(BUILD_SHARED_LIBS "Enable shared library build of MFEM" ON)
option(MFEM_USE_LIBUNWIND "Enable backtrace for errors." OFF)
option(MFEM_USE_MPI "Enable MPI parallel build" OFF)
option(MFEM_USE_METIS "Enable METIS usage" ${MFEM_USE_MPI})

set(MFEM_PRECISION "double" CACHE STRING
    "Floating-point precision to use: single, or double")

option(MFEM_ENABLE_TESTING "Enable the ctest framework for testing" ON)
option(MFEM_ENABLE_EXAMPLES "Build all of the examples" OFF)
option(MFEM_ENABLE_MINIAPPS "Build all of the miniapps" OFF)
option(MFEM_ENABLE_GOOGLE_BENCHMARKS "Build all of the Google benchmarks" OFF)
