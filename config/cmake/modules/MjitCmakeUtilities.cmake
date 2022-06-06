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

string(ASCII 27 ESC)

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

########################
# ADD MJIT EXECUTABLE  #
########################
function(add_mjit_executable)
    add_executable(mjit general/jit/parser.cpp)

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

    foreach (dir ${TPL_INCLUDE_DIRS})
      target_include_directories(mjit PRIVATE ${dir})
    endforeach (dir "${MFEM_INCLUDE_DIRS}")

    if (MFEM_USE_MPI)
        message(NOTICE "\t${ESC}[33m[MPI_CXX_COMPILER] ${MPI_CXX_COMPILER}${ESC}[m")
        set(MFEM_CXX ${MPI_CXX_COMPILER})
    endif(MFEM_USE_MPI)

    if(CMAKE_OSX_SYSROOT)
        message(NOTICE "\t${ESC}[33m[CMAKE_OSX_SYSROOT] ${CMAKE_OSX_SYSROOT}${ESC}[m")
        set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    endif(CMAKE_OSX_SYSROOT)

    if (MFEM_USE_CUDA)
       message("${ESC}[1;32m")
       string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
       message(NOTICE "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
       message(NOTICE "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
       message(NOTICE "CMAKE_CUDA_FLAGS_${BUILD_TYPE}: ${CMAKE_CUDA_FLAGS_${BUILD_TYPE}}")
       message(NOTICE "CUDA_FLAGS: ${CUDA_FLAGS}")
       message(NOTICE "CUDA_ARCH: ${CUDA_ARCH}")
       message(NOTICE "TPL_INCLUDE_DIRS: ${TPL_INCLUDE_DIRS}")
       message("${ESC}[m")

       set(MFEM_CXX ${CMAKE_CUDA_COMPILER})
       set(MFEM_EXT_LIBS "")
       set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -x=cu ${CUDA_FLAGS}")
       set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -arch=${CUDA_ARCH}")
       set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -ccbin ${CMAKE_CUDA_HOST_COMPILER}")
       string(REGEX REPLACE "[ ]+" " " MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS}")

       string(REPLACE "-x=cu" "" MFEM_LINK_FLAGS ${MFEM_BUILD_FLAGS})
       string(REPLACE "-xhip" "" MFEM_LINK_FLAGS ${MFEM_LINK_FLAGS})
       string(REGEX REPLACE "[ ]+" " " MFEM_LINK_FLAGS "${MFEM_LINK_FLAGS}")

       set_source_files_properties(general/jit/parser.cpp PROPERTIES LANGUAGE CUDA)
   endif(MFEM_USE_CUDA)

   message(NOTICE "\t${ESC}[33m[MFEM_CXX] ${MFEM_CXX}${ESC}[m")
   message(NOTICE "\t${ESC}[33m[CMAKE_CXX_COMPILER] ${CMAKE_CXX_COMPILER}${ESC}[m")
   message(NOTICE "\t${ESC}[33m[MFEM_EXT_LIBS] '${MFEM_EXT_LIBS}'${ESC}[m")
   message(NOTICE "\t${ESC}[33m[MFEM_BUILD_FLAGS] '${MFEM_BUILD_FLAGS}'${ESC}[m")
   message(NOTICE "\t${ESC}[33m[MFEM_LINK_FLAGS] '${MFEM_LINK_FLAGS}'${ESC}[m")

   target_compile_definitions(mjit PRIVATE
           "MFEM_CXX=\"${MFEM_CXX}\""
           "MFEM_EXT_LIBS=\"${MFEM_EXT_LIBS}\""
           "MFEM_BUILD_FLAGS=\"${MFEM_BUILD_FLAGS}\""
           "MFEM_LINK_FLAGS=\"${MFEM_LINK_FLAGS}\"")

    target_compile_definitions(mjit PRIVATE
      "MFEM_CONFIG_FILE=\"${PROJECT_BINARY_DIR}/config/_config.hpp\"")
endfunction(add_mjit_executable)

endif(MFEM_USE_JIT)
