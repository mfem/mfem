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

#add_library(sloshing_lib MODULE sloshing.c)
#set_target_properties(sloshing_lib PROPERTIES OUTPUT_NAME fun)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/potential/sloshing.c
               ${PROJECT_BINARY_DIR}/miniapps/potential/sloshing.c
               COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/miniapps/potential/airy.c
               ${PROJECT_BINARY_DIR}/miniapps/potential/airy.c
               COPYONLY)

add_mfem_miniapp(lin-wave1
  MAIN lin-wave1.cpp
  LIBRARIES ${CMAKE_DL_LIBS} mfem)

add_mfem_miniapp(lin-wave2
  MAIN lin-wave2.cpp
  LIBRARIES ${CMAKE_DL_LIBS} mfem)

add_mfem_miniapp(lin-wave3
  MAIN lin-wave3.cpp
  LIBRARIES ${CMAKE_DL_LIBS} mfem)
