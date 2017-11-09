// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mesh_headers.hpp"

namespace mfem
{

Geometry::Type GeometryType(Element::Type type)
{
   switch (type)
   {
      case Element::MIXED:
         return Geometry::MIXED;
      case Element::POINT:
         return Geometry::POINT;
      case Element::SEGMENT:
         return Geometry::SEGMENT;
      case Element::TRIANGLE:
         return Geometry::TRIANGLE;
      case Element::QUADRILATERAL:
         return Geometry::SQUARE;
      case Element::TETRAHEDRON:
         return Geometry::TETRAHEDRON;
      case Element::PRISM:
         return Geometry::PRISM;
      case Element::HEXAHEDRON:
         return Geometry::CUBE;
      default:
         return Geometry::INVALID;
   }
}

void Element::SetVertices(const int *ind)
{
   int i, n, *v;

   n = GetNVertices();
   v = GetVertices();

   for (i = 0; i < n; i++)
   {
      v[i] = ind[i];
   }
}

}
