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

Point::Point( const int *ind, int attr ) : Element(Geometry::POINT)
{
   attribute = attr;
   indices[0] = ind[0];
}

void Point::GetVertices( Array<int> &v ) const
{
   v.SetSize( 1 );
   v[0] = indices[0];
}

PointFiniteElement PointFE;

}
