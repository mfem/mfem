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
