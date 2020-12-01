// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

Quadrilateral::Quadrilateral( const int *ind, int attr )
   : Element(Geometry::SQUARE)
{
   attribute = attr;
   for (int i=0; i<4; i++)
   {
      indices[i] = ind[i];
   }
}

Quadrilateral::Quadrilateral( int ind1, int ind2, int ind3, int ind4,
                              int attr ) : Element(Geometry::SQUARE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
}

void Quadrilateral::SetVertices(const int *ind)
{
   for (int i=0; i<4; i++)
   {
      indices[i] = ind[i];
   }
}

void Quadrilateral::GetVertices( Array<int> &v ) const
{
   v.SetSize( 4 );
   for (int i=0; i<4; i++)
   {
      v[i] = indices[i];
   }
}

BiLinear2DFiniteElement QuadrilateralFE;

}
