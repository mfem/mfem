# Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

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
