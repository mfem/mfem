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

########################
# ADD MJIT EXECUTABLE  #
########################
function(add_mjit_executable)
    add_executable(mjit general/jit/parser.cpp)

    foreach (dir ${TPL_INCLUDE_DIRS})
       target_include_directories(mjit PRIVATE ${dir})
    endforeach (dir "${MFEM_INCLUDE_DIRS}")

    if(CMAKE_OSX_SYSROOT)
        set(MFEM_BUILD_FLAGS "${MFEM_BUILD_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
        set(MFEM_LINK_FLAGS "${MFEM_BUILD_FLAGS}")
    endif(CMAKE_OSX_SYSROOT)

    if (MFEM_USE_MPI)
      set(MFEM_CXX ${MPI_CXX_COMPILER})
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

    if (MFEM_USE_CUDA)
       set(MFEM_EXT_LIBS "")
       set(MFEM_CXX ${CMAKE_CUDA_COMPILER})
       set(MFEM_LINK_FLAGS "${MFEM_BUILD_FLAGS} -arch=${CUDA_ARCH} ${CUDA_FLAGS}")
       set(MFEM_LINK_FLAGS "${MFEM_LINK_FLAGS} -ccbin ${CMAKE_CUDA_HOST_COMPILER}")
       set(MFEM_BUILD_FLAGS "-x=cu ${MFEM_LINK_FLAGS}")
       set_source_files_properties(general/jit/parser.cpp PROPERTIES LANGUAGE CUDA)
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
           "MFEM_BUILD_FLAGS=\"${MFEM_BUILD_FLAGS}\"")

    target_compile_definitions(mjit PRIVATE
      "MFEM_CONFIG_FILE=\"${PROJECT_BINARY_DIR}/config/_config.hpp\"")
endfunction(add_mjit_executable)

endif(MFEM_USE_JIT)
