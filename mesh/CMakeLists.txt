# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

set(SRCS
  attribute_sets.cpp
  element.cpp
  exodus_writer.cpp
  face_nbr_geom.cpp
  gmsh.cpp
  hexahedron.cpp
  mesh.cpp
  mesh_operators.cpp
  mesh_readers.cpp
  ncmesh.cpp
  nurbs.cpp
  point.cpp
  pyramid.cpp
  quadrilateral.cpp
  segment.cpp
  spacing.cpp
  tetrahedron.cpp
  triangle.cpp
  vertex.cpp
  vtk.cpp
  wedge.cpp
  submesh/submesh.cpp
  submesh/ncsubmesh.cpp
  submesh/submesh_utils.cpp
  submesh/transfermap.cpp
  )

set(HDRS
  attribute_sets.hpp
  element.hpp
  face_nbr_geom.hpp
  gmsh.hpp
  hexahedron.hpp
  mesh.hpp
  mesh_headers.hpp
  mesh_operators.hpp
  ncmesh.hpp
  nurbs.hpp
  point.hpp
  pyramid.hpp
  quadrilateral.hpp
  segment.hpp
  spacing.hpp
  tetrahedron.hpp
  tmesh.hpp
  triangle.hpp
  vertex.hpp
  vtk.hpp
  wedge.hpp
  submesh/ncsubmesh.hpp
  submesh/submesh.hpp
  submesh/submesh_utils.hpp
  submesh/transfer_category.hpp
  submesh/transfermap.hpp
  )

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pmesh.cpp
    pncmesh.cpp
    submesh/pncsubmesh.cpp
    submesh/psubmesh.cpp
    submesh/ptransfermap.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # headers added all the time.
  list(APPEND HDRS
    pmesh.hpp
    pncmesh.hpp
    submesh/pncsubmesh.hpp
    submesh/psubmesh.hpp
    submesh/ptransfermap.hpp)
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
