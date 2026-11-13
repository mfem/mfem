// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NCMESH_FACE_PARTITION_HPP
#define MFEM_NCMESH_FACE_PARTITION_HPP

#include "../general/array.hpp"
#include "../general/globals.hpp"
#include "../fem/geom.hpp"

namespace mfem
{

/** A face partition in a common, oriented reference-face coordinate system.
    Cells are convex triangles or quadrilaterals. The caller supplies a complete,
    nonoverlapping partition of the same parent face on each side. */
struct NCFacePartition
{
   struct Point { real_t x, y; };
   struct Cell
   {
      Geometry::Type geometry;
      Point vertex[4];
   };

   Array<Cell> cells;
};

enum class NCFaceRelation
{
   Matching,
   FirstRefinesSecond,
   SecondRefinesFirst,
   Incompatible
};

// This is a geometric relation, not a claim that BuildFaceList can represent
// the resulting master/slave relation for every combination of cell types.
NCFaceRelation CompareNCFacePartitions(const NCFacePartition &first,
                                       const NCFacePartition &second);

} // namespace mfem

#endif // MFEM_NCMESH_FACE_PARTITION_HPP
