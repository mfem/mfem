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
    MAIN automata.cpp
    LIBRARIES mfem mfem_miniapps_common)

add_mfem_miniapp(toy_life
    MAIN life.cpp
    LIBRARIES mfem mfem_miniapps_common)

add_mfem_miniapp(toy_mandel
    MAIN mandel.cpp
    LIBRARIES mfem mfem_miniapps_common)

add_mfem_miniapp(toy_rubik
    MAIN rubik.cpp
    LIBRARIES mfem mfem_miniapps_common)

add_mfem_miniapp(toy_snake
    MAIN snake.cpp
    LIBRARIES mfem mfem_miniapps_common)