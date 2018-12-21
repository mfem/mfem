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
  array.cpp
  cuda.cpp
  error.cpp
  config.cpp
  globals.cpp
  gzstream.cpp
  isockstream.cpp
  mm.cpp
  occa.cpp
  okina.cpp
  optparser.cpp
  osockstream.cpp
  sets.cpp
  socketstream.cpp
  stable3d.cpp
  table.cpp
  tic_toc.cpp
  version.cpp
  )

list(APPEND HDRS
  array.hpp
  binaryio.hpp
  config.hpp
  cuda.hpp
  error.hpp
  globals.hpp
  gzstream.hpp
  hash.hpp
  isockstream.hpp
  karray.hpp
  mem_alloc.hpp
  mm.hpp
  occa.hpp
  okina.hpp
  optparser.hpp
  osockstream.hpp
  sets.hpp
  socketstream.hpp
  sort_pairs.hpp
  stable3d.hpp
  table.hpp
  tassign.hpp
  tic_toc.hpp
  text.hpp
  version.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS communication.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # header added all the time.
  list(APPEND HDRS communication.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
