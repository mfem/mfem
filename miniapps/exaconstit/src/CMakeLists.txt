#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(EXACONSTIT_HEADERS
    ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h
    BCData.hpp
    BCManager.hpp
    mechanics_coefficient.hpp
    mechanics_integrators.hpp
    mechanics_operator.hpp
    mechanics_solver.hpp
    option_parser.hpp
    userumat.h
    ./TOML_Reader/cpptoml.h
    )

set(EXACONSTIT_SOURCES
    BCData.cpp
    BCManager.cpp
    mechanics_coefficient.cpp
    mechanics_integrators.cpp
    mechanics_operator.cpp
    mechanics_solver.cpp
    option_parser.cpp
    ./umat_tests/userumat.cxx
    ./umat_tests/umat.f
    )

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(EXACONSTIT_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_DEPENDS
                             DEPENDS_ON  mfem ecmech raja mpi hypre snls metis)

#include_directories(BEFORE ${PROJECT_BINARY_DIR})

#------------------------------------------------------------------------------
# Defines
#------------------------------------------------------------------------------
set(EXACONSTIT_DEFINES HAVE_EXACONSTIT)

SET(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
set(EXACONSTIT_INCLUDES ${CMAKE_SOURCE_DIR}/src/interface
                        ${HEADER_INCLUDE_DIR}
                        ${CMAKE_SOURCE_DIR}/src/)
#------------------------------------------------------------------------------
# Build Targets
#------------------------------------------------------------------------------

blt_add_library(NAME    exaconstit_static
                        OUTPUT_NAME exaconstit
                        DEFINES     ${EXACONSTIT_DEFINES}
                        INCLUDES    ${EXACONSTIT_INCLUDES}
                        DEPENDS_ON  ${EXACONSTIT_DEPENDS}
                        HEADERS     ${EXACONSTIT_HEADERS}
                        SOURCES     ${EXACONSTIT_SOURCES}
			OBJECT      TRUE
                        SHARED      FALSE)

#------------------------------------------------------------------------------
# Install files
#------------------------------------------------------------------------------

#set(EXACONSTIT_INSTALLED_HEADERS
#    ${EXACONSTIT_HEADERS})

#install(FILES ${EXACONSTIT_INSTALLED_HEADERS} DESTINATION include)

#install(TARGETS exaconstit_static DESTINATION lib)


blt_add_executable(NAME       mechanics
                   SOURCES    mechanics_driver.cpp
                   OUTPUT_DIR ${BINARY_DIR}
                   DEPENDS_ON exaconstit_static)
