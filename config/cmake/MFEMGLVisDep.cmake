# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Prefer an imported target. Otherwise use the first variable that names a
# library. Older Find modules set only the variables. FindGLVis.cmake and
# MFEMConfig.cmake.in both include this file.
if(NOT COMMAND mfem_glvis_dep)
    macro(mfem_glvis_dep OUT)
        set(${OUT} "")
        foreach(item ${ARGN})
            if(TARGET "${item}")
                set(${OUT} "${item}")
                break()
            endif()
            if(DEFINED ${item} AND NOT "${${item}}" STREQUAL ""
               AND NOT "${${item}}" MATCHES "NOTFOUND")
                set(${OUT} "${${item}}")
                break()
            endif()
        endforeach()
        if("${${OUT}}" STREQUAL "")
            message(FATAL_ERROR
                "GLVis dependency not found. Tried: ${ARGN}")
        endif()
    endmacro()
endif()
