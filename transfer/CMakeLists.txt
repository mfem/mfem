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

list(APPEND SRCS
  cut.cpp
  transferutils.cpp
  mortarassemble.cpp
  mortarassembler.cpp
  mortarintegrator.cpp
  pmortarassembler.cpp
  )

list(APPEND HDRS
  cut.hpp
  transferutils.hpp
  mortarassemble.hpp
  mortarassembler.hpp
  mortarintegrator.hpp
  pmortarassembler.hpp
  transfer.hpp
  )


convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
