// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "mesh_headers.hpp"

namespace mfem
{

Segment::Segment( const int *ind, int attr ) : Element(Geometry::SEGMENT)
{
   attribute = attr;
   for (int i=0; i<2; i++)
   {
      indices[i] = ind[i];
   }
}

Segment::Segment( int ind1, int ind2, int attr ) : Element(Geometry::SEGMENT)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
}

void Segment::SetVertices(const int *ind)
{
   indices[0] = ind[0];
   indices[1] = ind[1];
}

void Segment::GetVertices(Array<int> &v) const
{
   v.SetSize(2);
   std::copy(indices, indices + 2, v.begin());
}

void Segment::SetVertices(const Array<int> &v)
{
   MFEM_ASSERT(v.Size() == 2, "!");
   std::copy(v.begin(), v.end(), indices);
}

Linear1DFiniteElement SegmentFE;

}
