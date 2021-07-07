##################################################################################
#
#                    Set defaults for XSDK CMake projects
#
##################################################################################

#
# This module implements standard behavior for XSDK CMake projects.  The main
# thing it does in XSDK mode (i.e. USE_XSDK_DEFAULTS=TRUE) is to print out
# when the env vars CC, CXX, FC and compiler flags CFLAGS, CXXFLAGS, and
# FFLAGS/FCFLAGS are used to select the compilers and compiler flags (raw
# CMake does this silently) and to set BUILD_SHARED_LIBS=TRUE and
# CMAKE_BUILD_TYPE=DEBUG by default.  It does not implement *all* of the
# standard XSDK configuration parameters.  The parent CMake project must do
# that.
#
# Note that when USE_XSDK_DEFAULTS=TRUE, then the Fortran flags will be read
# from either of the env vars FFLAGS or FCFLAGS.  If both are set, but are the
# same, then FFLAGS it used (which is the same as FCFLAGS).  However, if both
# are set but are not equal, then a FATAL_ERROR is raised and CMake configure
# processing is stopped.
#
# To be used in a parent project, this module must be included after
#
#   PROJECT(${PROJECT_NAME}  NONE)
#
# is called but before the compilers are defined and processed using:
#
#   ENABLE_LANGUAGE(<LANG>)
#
# For example, one would do:
#
#   PROJECT(${PROJECT_NAME}  NONE)
#   ...
#   SET(USE_XSDK_DEFAULTS_DEFAULT TRUE) # Set to false if desired
#   INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/stdk/XSDKDefaults.cmake")
#   ...
#   ENABLE_LANGUAGE(C)
#   ENABLE_LANGUAGE(C++)
#   ENABLE_LANGUAGE(Fortran)
#
# The variable `USE_XSDK_DEFAULTS_DEFAULT` is used as the default for the
# cache var `USE_XSDK_DEFAULTS`.  That way, a project can decide if it wants
# XSDK defaults turned on or off by default and users can independently decide
# if they want the CMake project to use standard XSDK behavior or raw CMake
# behavior.
#
# By default, the XSDKDefaults.cmake module assumes that the project will need
# C, C++, and Fortran.  If any language is not needed then, set
# XSDK_ENABLE_C=OFF, XSDK_ENABLE_CXX=OFF, or XSDK_ENABLE_Fortran=OFF *before*
# including this module.  Note, these variables are *not* cache vars because a
# project either does or does not have C, C++ or Fortran source files, the
# user has nothing to do with this so there is no need for cache vars.  The
# parent CMake project just needs to tell XSDKDefault.cmake what languages is
# needs or does not need.
#
# For example, if the parent CMake project only needs C, then it would do:
#
#   PROJECT(${PROJECT_NAME}  NONE)'
#   ...
#   SET(USE_XSDK_DEFAULTS_DEFAULT TRUE)
#   SET(XSDK_ENABLE_CXX OFF)
#   SET(XSDK_ENABLE_Fortran OFF)
#   INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/stdk/XSDKDefaults.cmake")
#   ...
#   ENABLE_LANGAUGE(C)
#
# This module code will announce when it sets any variables.
#

#
# Helper functions
#

IF (NOT COMMAND PRINT_VAR)
  FUNCTION(PRINT_VAR  VAR_NAME)
    MESSAGE(STATUS "${VAR_NAME} = '${${VAR_NAME}}'")
  ENDFUNCTION()
ENDIF()

IF (NOT COMMAND SET_DEFAULT)
  MACRO(SET_DEFAULT VAR)
    IF ("${${VAR}}" STREQUAL "")
      SET(${VAR} ${ARGN})
    ENDIF()
  ENDMACRO()
ENDIF()

#
# XSDKDefaults.cmake control variables
#

# USE_XSDK_DEFAULTS
IF ("${USE_XSDK_DEFAULTS_DEFAULT}" STREQUAL "")
  SET(USE_XSDK_DEFAULTS_DEFAULT  FALSE)
ENDIF()
SET(USE_XSDK_DEFAULTS  ${USE_XSDK_DEFAULTS_DEFAULT}  CACHE  BOOL
  "Use XSDK defaults and behavior.")
PRINT_VAR(USE_XSDK_DEFAULTS)

SET_DEFAULT(XSDK_ENABLE_C  TRUE)
SET_DEFAULT(XSDK_ENABLE_CXX  TRUE)
SET_DEFAULT(XSDK_ENABLE_Fortran  TRUE)

# Handle the compiler and flags for a language
MACRO(XSDK_HANDLE_LANG_DEFAULTS  CMAKE_LANG_NAME  ENV_LANG_NAME
  ENV_LANG_FLAGS_NAMES
  )

  # Announce using env var ${ENV_LANG_NAME}
  IF (NOT "$ENV{${ENV_LANG_NAME}}" STREQUAL "" AND
    "${CMAKE_${CMAKE_LANG_NAME}_COMPILER}" STREQUAL ""
    )
    MESSAGE("-- " "XSDK: Setting CMAKE_${CMAKE_LANG_NAME}_COMPILER from env var"
      " ${ENV_LANG_NAME}='$ENV{${ENV_LANG_NAME}}'!")
    SET(CMAKE_${CMAKE_LANG_NAME}_COMPILER "$ENV{${ENV_LANG_NAME}}" CACHE FILEPATH
      "XSDK: Set by default from env var ${ENV_LANG_NAME}")
  ENDIF()

  # Announce using env var ${ENV_LANG_FLAGS_NAME}
  FOREACH(ENV_LANG_FLAGS_NAME  ${ENV_LANG_FLAGS_NAMES})
    IF (NOT "$ENV{${ENV_LANG_FLAGS_NAME}}" STREQUAL "" AND
      "${CMAKE_${CMAKE_LANG_NAME}_FLAGS}" STREQUAL ""
      )
      MESSAGE("-- " "XSDK: Setting CMAKE_${CMAKE_LANG_NAME}_FLAGS from env var"
        " ${ENV_LANG_FLAGS_NAME}='$ENV{${ENV_LANG_FLAGS_NAME}}'!")
      SET(CMAKE_${CMAKE_LANG_NAME}_FLAGS "$ENV{${ENV_LANG_FLAGS_NAME}} " CACHE  STRING
        "XSDK: Set by default from env var ${ENV_LANG_FLAGS_NAME}")
      # NOTE: CMake adds the space after $ENV{${ENV_LANG_FLAGS_NAME}} so we
      # duplicate that here!
    ENDIF()
  ENDFOREACH()

ENDMACRO()


#
# Set XSDK Defaults
#

# Set default compilers and flags
IF (USE_XSDK_DEFAULTS)

  # Handle env vars for languages C, C++, and Fortran

  IF (XSDK_ENABLE_C)
    XSDK_HANDLE_LANG_DEFAULTS(C  CC  CFLAGS)
  ENDIF()

  IF (XSDK_ENABLE_CXX)
    XSDK_HANDLE_LANG_DEFAULTS(CXX  CXX  CXXFLAGS)
  ENDIF()

  IF (XSDK_ENABLE_Fortran)
    SET(ENV_FFLAGS "$ENV{FFLAGS}")
    SET(ENV_FCFLAGS "$ENV{FCFLAGS}")
    IF (
      (NOT "${ENV_FFLAGS}" STREQUAL "") AND (NOT "${ENV_FCFLAGS}" STREQUAL "")
      AND
      ("${CMAKE_Fortran_FLAGS}" STREQUAL "")
      )
      IF (NOT "${ENV_FFLAGS}" STREQUAL "${ENV_FCFLAGS}")
        MESSAGE(FATAL_ERROR "Error, env vars FFLAGS='${ENV_FFLAGS}' and"
          " FCFLAGS='${ENV_FCFLAGS}' are both set in the env but are not equal!")
      ENDIF()
    ENDIF()
    XSDK_HANDLE_LANG_DEFAULTS(Fortran  FC  "FFLAGS;FCFLAGS")
  ENDIF()

  # Set XSDK defaults for other CMake variables

  IF ("${BUILD_SHARED_LIBS}"  STREQUAL  "")
    MESSAGE("-- " "XSDK: Setting default BUILD_SHARED_LIBS=TRUE")
    SET(BUILD_SHARED_LIBS  TRUE  CACHE  BOOL  "Set by default in XSDK mode")
  ENDIF()

  IF ("${CMAKE_BUILD_TYPE}"  STREQUAL  "")
    MESSAGE("-- " "XSDK: Setting default CMAKE_BUILD_TYPE=DEBUG")
    SET(CMAKE_BUILD_TYPE  DEBUG  CACHE  STRING  "Set by default in XSDK mode")
  ENDIF()

ENDIF()


##################################################################################
#
#               MFEM-specific additions: set TPL MFEM_USE_* defaults
#
##################################################################################

IF (DEFINED TPL_ENABLE_MPI)
  SET(MFEM_USE_MPI ${TPL_ENABLE_MPI} CACHE BOOL "Enable MPI parallel build" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_METIS)
  SET(MFEM_USE_METIS ${TPL_ENABLE_METIS} CACHE BOOL "Enable METIS usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_ZLIB)
  SET(MFEM_USE_ZLIB ${TPL_ENABLE_ZLIB} CACHE BOOL "Enable zlib for compressed data streams." FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_LIBUNWIND)
  SET(MFEM_USE_LIBUNWIND ${TPL_ENABLE_LIBUNWIND} CACHE BOOL "Enable backtrace for errors." FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_LAPACK)
  SET(MFEM_USE_LAPACK ${TPL_ENABLE_LAPACK} CACHE BOOL "Enable LAPACK usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_SUNDIALS)
  SET(MFEM_USE_SUNDIALS ${TPL_ENABLE_SUNDIALS} CACHE BOOL "Enable SUNDIALS usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_MESQUITE)
  SET(MFEM_USE_MESQUITE ${TPL_ENABLE_MESQUITE} CACHE BOOL "Enable MESQUITE usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_SUITESPARSE)
  SET(MFEM_USE_SUITESPARSE ${TPL_ENABLE_SUITESPARSE} CACHE BOOL "Enable SuiteSparse usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_SUPERLU)
  SET(MFEM_USE_SUPERLU ${TPL_ENABLE_SUPERLU} CACHE BOOL "Enable SuperLU_DIST usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_STRUMPACK)
  SET(MFEM_USE_STRUMPACK ${TPL_ENABLE_STRUMPACK} CACHE BOOL "Enable STRUMPACK usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_GINKGO)
  SET(MFEM_USE_GINKGO ${TPL_ENABLE_GINKGO} CACHE BOOL "Enable GINKGO usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_GNUTLS)
  SET(MFEM_USE_GNUTLS ${TPL_ENABLE_GNUTLS} CACHE BOOL "Enable GNUTLS usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_NETCDF)
  SET(MFEM_USE_NETCDF ${TPL_ENABLE_NETCDF} CACHE BOOL "Enable NETCDF usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_PETSC)
  SET(MFEM_USE_PETSC ${TPL_ENABLE_PETSC} CACHE BOOL "Enable PETSc support." FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_SLEPC)
  SET(MFEM_USE_SLEPC ${TPL_ENABLE_SLEPC} CACHE BOOL "Enable SLEPc support." FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_MPFR)
  SET(MFEM_USE_MPFR ${TPL_ENABLE_MPFR} CACHE BOOL "Enable MPFR usage." FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_SIDRE)
  SET(MFEM_USE_SIDRE ${TPL_ENABLE_SIDRE} CACHE BOOL "Enable Axom/Sidre usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_CONDUIT)
  SET(MFEM_USE_CONDUIT ${TPL_ENABLE_CONDUIT} CACHE BOOL "Enable Conduit usage" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_PUMI)
  SET(MFEM_USE_PUMI ${TPL_ENABLE_PUMI} CACHE BOOL "Enable PUMI" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_CUDA)
  SET(MFEM_USE_CUDA ${TPL_ENABLE_CUDA} CACHE BOOL "Enable CUDA" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_OCCA)
  SET(MFEM_USE_OCCA ${TPL_ENABLE_OCCA} CACHE BOOL "Enable OCCA" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_RAJA)
  SET(MFEM_USE_RAJA ${TPL_ENABLE_RAJA} CACHE BOOL "Enable RAJA" FORCE)
ENDIF()

IF (DEFINED TPL_ENABLE_UMPIRE)
  SET(MFEM_USE_UMPIRE ${TPL_ENABLE_UMPIRE} CACHE BOOL "Enable Umpire" FORCE)
ENDIF()
