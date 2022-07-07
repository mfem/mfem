# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_JIT)

#################################
# set_mjit_sources_dependencies #
#################################
function(set_mjit_sources_dependencies TARGET SOURCES)
    add_custom_target(${TARGET})
    # 'mjit' all input files from source to binary directory
    foreach(source IN LISTS ${SOURCES})
        get_filename_component(name ${source} NAME)
        get_filename_component(dir ${source} DIRECTORY)
        file(RELATIVE_PATH source_path ${CMAKE_CURRENT_SOURCE_DIR} ${dir})
        set(binary_path ${CMAKE_CURRENT_BINARY_DIR}/${relpath})
        set(jit ${CMAKE_CURRENT_BINARY_DIR}/${source_path}/${name})
        add_custom_command(OUTPUT ${jit}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${source_path}
            COMMAND mjit ${source} -o ${jit} DEPENDS mjit ${source})
        set(${TARGET} ${${TARGET}} ${jit})
        # create the dependency name from source_path and name${TARGET}
        string(REPLACE " " "_" source_d ${TARGET}/${source_path}/${name})
        string(REPLACE "." "_" source_d ${source_d})
        string(REPLACE "-" "_" source_d ${source_d})
        string(REPLACE "/" "_" source_d ${source_d})
        add_custom_target(${source_d} DEPENDS ${jit})
        add_dependencies(${TARGET} ${source_d})
        set_source_files_properties(${jit} PROPERTIES COMPILE_OPTIONS -I${dir})
        if (MFEM_USE_CUDA)
           set_source_files_properties(${jit} PROPERTIES LANGUAGE CUDA)
       endif(MFEM_USE_CUDA)
    endforeach()
    set(${TARGET} ${${TARGET}} PARENT_SCOPE)
endfunction(set_mjit_sources_dependencies)

######################
# MFEM JIT CONFIGURE #
######################
function(mfem_mjit_configure)
    add_executable(mjit general/jit/parser.cpp)
    #message(info "[JIT] old MFEM_BUILD_FLAGS: ${MFEM_BUILD_FLAGS}")

    string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
    #message(info "[JIT] CUDA_FLAGS: ${CUDA_FLAGS}")
    #message(info "[JIT] CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    #message(info "[JIT] CMAKE_CUDA_FLAGS_${BUILD_TYPE}: ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
    set(MFEM_BUILD_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
    set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -std=c++${CMAKE_CUDA_STANDARD}")
    #message(info "[JIT] MFEM_BUILD_FLAGS: ${MFEM_BUILD_FLAGS}")

    set(MFEM_TPLFLAGS "")
    foreach (dir ${TPL_INCLUDE_DIRS})
       target_include_directories(mjit PRIVATE ${dir})
       set(MFEM_TPLFLAGS "${MFEM_TPLFLAGS} -I${dir}")
    endforeach (dir "${TPL_INCLUDE_DIRS}")

    if(CMAKE_OSX_SYSROOT)
        set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
        set(MFEM_LINK_FLAGS "${MFEM_BUILD_FLAGS}")
    endif(CMAKE_OSX_SYSROOT)

    if (MFEM_USE_MPI)
      if (MPI_CXX_INCLUDE_PATH)
        target_include_directories(mjit PRIVATE "${MPI_CXX_INCLUDE_PATH}")
      endif(MPI_CXX_INCLUDE_PATH)
      if (MPI_CXX_COMPILE_FLAGS)
        target_compile_options(mjit PRIVATE ${MPI_CXX_COMPILE_ARGS})
      endif(MPI_CXX_COMPILE_FLAGS)
      if (MPI_CXX_LINK_FLAGS)
        set_target_properties(mjit PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
      endif(MPI_CXX_LINK_FLAGS)
    endif(MFEM_USE_MPI)

    set(MFEM_XLINKER "-Wl,")

    if (MFEM_USE_CUDA)
       set(MFEM_EXT_LIBS "")
       set(MFEM_CXX ${CMAKE_CUDA_COMPILER})
       set(MFEM_LINK_FLAGS "${MFEM_BUILD_FLAGS}")
       if (MFEM_USE_MPI)
          set(MFEM_LINK_FLAGS "${MFEM_LINK_FLAGS} -ccbin ${MPI_CXX_COMPILER}")
       else(MFEM_USE_MPI)
          set(MFEM_LINK_FLAGS "${MFEM_LINK_FLAGS} -ccbin ${CMAKE_CUDA_HOST_COMPILER}")
       endif(MFEM_USE_MPI)
       set(MFEM_BUILD_FLAGS "-x cu ${MFEM_LINK_FLAGS}")
       set_source_files_properties(general/jit/parser.cpp PROPERTIES LANGUAGE CUDA)
       set(MFEM_XCOMPILER "-Xcompiler=")
       set(MFEM_XLINKER "-Xlinker=")
   endif(MFEM_USE_CUDA)

   if (MFEM_USE_HIP)
      set(MFEM_EXT_LIBS "")
      set(MFEM_CXX ${HIP_HIPCC_EXECUTABLE})
      set(MFEM_LINK_FLAGS "${MFEM_BUILD_FLAGS} --offload-arch=${HIP_ARCH}")
      set(MFEM_BUILD_FLAGS "-x hip ${MFEM_LINK_FLAGS}")
   endif(MFEM_USE_HIP)

   target_compile_definitions(mjit PRIVATE
           "MFEM_CXX=\"${MFEM_CXX}\""
           "MFEM_EXT_LIBS=\"${MFEM_EXT_LIBS}\""
           "MFEM_LINK_FLAGS=\"${MFEM_LINK_FLAGS}\""
           "MFEM_BUILD_FLAGS=\"${MFEM_BUILD_FLAGS} ${MFEM_TPLFLAGS}\"")

    target_compile_definitions(mjit PRIVATE
           "MFEM_CONFIG_FILE=\"${PROJECT_BINARY_DIR}/config/_config.hpp\"")

    if (APPLE)
        set(MFEM_SO_PREFIX "-all_load")
        set(MFEM_SO_POSTFIX "")
        set(MFEM_INSTALL_BACKUP "")
    else(APPLE)
        set(MFEM_SO_PREFIX "${MFEM_XLINKER}--whole-archive")
        set(MFEM_SO_POSTFIX "${MFEM_XLINKER}--no-whole-archive")
        set(MFEM_INSTALL_BACKUP "--backup=none")
    endif(APPLE)

    # CMAKE_SHARED_LIBRARY_SUFFIX has an extra "." prefix to remove
    string(REPLACE "." "" MFEM_SO_EXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set_property(SOURCE general/jit/jit.cpp
                 PROPERTY COMPILE_DEFINITIONS
                 MFEM_SO_EXT="${MFEM_SO_EXT}"
                 MFEM_PICFLAG="${MFEM_XCOMPILER}${CMAKE_SHARED_LIBRARY_CXX_FLAGS}"
                 MFEM_XCOMPILER="${MFEM_XCOMPILER}"
                 MFEM_XLINKER="${MFEM_XLINKER}"
                 MFEM_AR="ar"
                 MFEM_INSTALL_BACKUP="${MFEM_INSTALL_BACKUP}"
                 MFEM_SO_PREFIX="${MFEM_SO_PREFIX}"
                 MFEM_SO_POSTFIX="${MFEM_SO_POSTFIX}")

endfunction(mfem_mjit_configure)

endif(MFEM_USE_JIT)
