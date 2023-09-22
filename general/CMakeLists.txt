# Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

list(APPEND SRCS
  array.cpp
  binaryio.cpp
  cuda.cpp
  device.cpp
  error.cpp
  gecko.cpp
  globals.cpp
  hash.cpp
  isockstream.cpp
  mem_manager.cpp
  occa.cpp
  optparser.cpp
  osockstream.cpp
  sets.cpp
  socketstream.cpp
  stable3d.cpp
  table.cpp
  tic_toc.cpp
  tinyxml2.cpp
  version.cpp
  hip.cpp
  )

list(APPEND HDRS
  annotation.hpp
  array.hpp
  backends.hpp
  binaryio.hpp
  cuda.hpp
  device.hpp
  error.hpp
  gecko.hpp
  globals.hpp
  zstr.hpp
  hash.hpp
  isockstream.hpp
  mdarray.hpp
  mdspan.hpp
  mem_alloc.hpp
  mem_manager.hpp
  occa.hpp
  forall.hpp
  optparser.hpp
  osockstream.hpp
  sets.hpp
  socketstream.hpp
  sort_pairs.hpp
  stable3d.hpp
  table.hpp
  tassign.hpp
  tic_toc.hpp
  tinyxml2.h
  text.hpp
  version.hpp
  hip.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS communication.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # header added all the time.
  list(APPEND HDRS communication.hpp)
endif()

if (MFEM_USE_ADIOS2)
  list(APPEND SRCS adios2stream.cpp)
  list(APPEND HDRS adios2stream.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
