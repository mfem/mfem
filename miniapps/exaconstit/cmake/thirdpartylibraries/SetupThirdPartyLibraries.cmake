# Provide backwards compatibility for *_PREFIX options
set(_tpls 
    mfem
    exacmech)

foreach(_tpl ${_tpls})
    string(TOUPPER ${_tpl} _uctpl)
    if (${_uctpl}_PREFIX)
        set(${_uctpl}_DIR ${${_uctpl}_PREFIX} CACHE PATH "")
        mark_as_advanced(${_uctpl}_PREFIX)
    endif()
endforeach()

################################
# MFEM
################################

if (DEFINED MFEM_DIR)
    include(cmake/thirdpartylibraries/FindMFEM.cmake)
    if (MFEM_FOUND)
        blt_register_library( NAME       mfem
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${MFEM_INCLUDE_DIRS}
                              LIBRARIES  ${MFEM_LIBRARY})
        #set(BLT_EXE_LINKER_FLAGS "${BLT_EXE_LINKER_FLAGS} ${MFEM_EXT_LIBS}" CACHE STRING "" FORCE)
	#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MFEM_EXT_LIBS}" CACHE STRING "" FORCE)
    else()
        message(FATAL_ERROR "Unable to find MFEM with given path ${MFEM_DIR}")
    endif()
else()
    message(FATAL_ERROR "MFEM_DIR was not provided. It is needed to find MFEM.")
endif()


################################
# ExaCMech
################################

if (DEFINED ECMECH_DIR)
    include(cmake/thirdpartylibraries/FindECMech.cmake)
    if (ECMECH_FOUND)
        blt_register_library( NAME       ecmech
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${ECMECH_INCLUDE_DIRS}
                              LIBRARIES  ${ECMECH_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find ExaCMech with given path ${ECMECH_DIR}")
    endif()
else()
    message(FATAL_ERROR "ECMECH_DIR was not provided. It is needed to find ExaCMech.")
endif()

################################
# RAJA
################################

if (DEFINED RAJA_DIR)
    include(cmake/thirdpartylibraries/FindRAJA.cmake)
    if (RAJA_FOUND)
        blt_register_library( NAME       raja
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${RAJA_INCLUDE_DIRS}
                              LIBRARIES  ${RAJA_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find RAJA with given path ${RAJA_DIR}")
    endif()
else()
    message(FATAL_ERROR "RAJA_DIR was not provided. It is needed to find RAJA.")
endif()

################################
# SNLS
################################

if (DEFINED SNLS_DIR)
    include(cmake/thirdpartylibraries/FindSNLS.cmake)
    if (SNLS_FOUND)
        blt_register_library( NAME       snls
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${SNLS_INCLUDE_DIRS}
                              LIBRARIES  ${SNLS_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find SNLS with given path ${SNLS_DIR}")
    endif()
else()
    message(FATAL_ERROR "SNLS_DIR was not provided. It is needed to find SNLS.")
endif()

################################
# HYPRE
################################

if (DEFINED HYPRE_DIR)
    include(cmake/thirdpartylibraries/FindHypre.cmake)
    if (HYPRE_FOUND)
        blt_register_library( NAME       hypre
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${HYPRE_INCLUDE_DIRS}
                              LIBRARIES  ${HYPRE_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find HYPRE with given path ${HYPRE_DIR}")
    endif()
else()
    message(FATAL_ERROR "HYPRE_DIR was not provided. It is needed to find HYPRE.")
endif()

################################
# METIS
################################

if (DEFINED METIS_DIR)
    include(cmake/thirdpartylibraries/FindMetis.cmake)
    if (METIS_FOUND)
        blt_register_library( NAME       metis
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${METIS_INCLUDE_DIRS}
                              LIBRARIES  ${METIS_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find METIS with given path ${METIS_DIR}")
    endif()
else()
    message(FATAL_ERROR "METIS_DIR was not provided. It is needed to find METIS.")
endif()

################################
# CONDUIT
################################

if (DEFINED CONDUIT_DIR)
    include(cmake/thirdpartylibraries/FindConduit.cmake)
    if (CONDUIT_FOUND)
        blt_register_library( NAME       conduit
                              TREAT_INCLUDES_AS_SYSTEM ON
                              INCLUDES   ${CONDUIT_INCLUDE_DIRS}
                              LIBRARIES  ${CONDUIT_LIBRARIES} ${CONDUIT_BLUEPRINT_LIBRARY} ${CONDUIT_RELAY_LIBRARY})
    else()
        message(FATAL_ERROR "Unable to find CONDUIT with given path ${CONDUIT_DIR}")
    endif()
else()
    message(FATAL_ERROR "CONDUIT_DIR was not provided. It is needed to find CONDUIT.")
endif()