################################
# Version
################################
set(PACKAGE_BUGREPORT "carson16@llnl.gov")

set(EXACONSTIT_VERSION_MAJOR 0)
set(EXACONSTIT_VERSION_MINOR 2)
set(EXACONSTIT_VERSION_PATCH \"0\")

set(HEADER_INCLUDE_DIR
    ${PROJECT_BINARY_DIR}/include
    CACHE PATH
    "Directory where all generated headers will go in the build tree")

configure_file( src/ExaConstit_Version.h.in
                ${HEADER_INCLUDE_DIR}/ExaConstit_Version.h )

################################
# Setup build options and their default values
################################
include(cmake/ExaConstitOptions.cmake)

################################
# Third party library setup
################################
include(cmake/thirdpartylibraries/SetupThirdPartyLibraries.cmake)

##------------------------------------------------------------------------------
## exaconstit_fill_depends_on_list
##
## This macro adds a dependency to the list if ENABLE_<dep name> or <dep name>_FOUND
##------------------------------------------------------------------------------
macro(exaconstit_fill_depends_list)

    set(options )
    set(singleValueArgs LIST_NAME)
    set(multiValueArgs DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT arg_LIST_NAME)
        message(FATAL_ERROR "exaconstit_fill_depends_list requires argument LIST_NAME")
    endif()

    if (NOT arg_DEPENDS_ON)
        message(FATAL_ERROR "exaconstit_fill_depends_list requires argument DEPENDS_ON")
    endif()

    foreach( _dep ${arg_DEPENDS_ON})
        string(TOUPPER ${_dep} _ucdep)

        if (ENABLE_${_ucdep} OR ${_ucdep}_FOUND)
            list(APPEND ${arg_LIST_NAME} ${_dep})
        endif()
    endforeach()

    unset(_dep)
    unset(_ucdep)

endmacro(exaconstit_fill_depends_list)
