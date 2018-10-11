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

set(SRCS
  element.cpp
  hexahedron.cpp
  mesh.cpp
  mesh_operators.cpp
  mesh_readers.cpp
  ncmesh.cpp
  nurbs.cpp
  point.cpp
  quadrilateral.cpp
  segment.cpp
  tetrahedron.cpp
  triangle.cpp
  vertex.cpp
  wedge.cpp
  )

set(HDRS
  element.hpp
  hexahedron.hpp
  mesh.hpp
  mesh_headers.hpp
  mesh_operators.hpp
  ncmesh.hpp
  nurbs.hpp
  point.hpp
  quadrilateral.hpp
  segment.hpp
  tetrahedron.hpp
  tmesh.hpp
  triangle.hpp
  vertex.hpp
  wedge.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pmesh.cpp
    pncmesh.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # headers added all the time.
  list(APPEND HDRS
    pmesh.hpp
    pncmesh.hpp)
endif()

if (MFEM_USE_MESQUITE)
  list(APPEND SRCS mesquite.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # header added all the time.
  list(APPEND HDRS mesquite.hpp)
endif()

if (MFEM_USE_PUMI)
  list(APPEND SRCS
    pumi.cpp)
  list(APPEND HDRS
    pumi.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
