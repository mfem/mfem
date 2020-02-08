# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

add_mfem_miniapp(toy_automata
    MAIN automata.cpp LIBRARIES mfem)

add_mfem_miniapp(toy_life
    MAIN life.cpp LIBRARIES mfem)

add_mfem_miniapp(toy_mandel
    MAIN mandel.cpp LIBRARIES mfem)

add_mfem_miniapp(toy_rubik
    MAIN rubik.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(toy_snake
    MAIN snake.cpp
    ${MFEM_MINIAPPS_COMMON_HEADERS}
    LIBRARIES mfem mfem-common)

add_mfem_miniapp(toy_lissajous
    MAIN lissajous.cpp LIBRARIES mfem)

add_mfem_miniapp(toy_mondrian
    MAIN mondrian.cpp LIBRARIES mfem)

if (MFEM_USE_CUDA)
   set_property(SOURCE automata.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE life.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE mandel.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE rubik.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE snake.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE lissajous.cpp PROPERTY LANGUAGE CUDA)
   set_property(SOURCE mondrian.cpp PROPERTY LANGUAGE CUDA)
endif()
