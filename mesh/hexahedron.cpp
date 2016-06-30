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

Hexahedron::Hexahedron(const int *ind, int attr)
   : Element(Geometry::CUBE)
{
   attribute = attr;
   for (int i = 0; i < 8; i++)
   {
      indices[i] = ind[i];
   }
}

Hexahedron::Hexahedron(int ind1, int ind2, int ind3, int ind4,
                       int ind5, int ind6, int ind7, int ind8,
                       int attr) : Element(Geometry::CUBE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;
   indices[5] = ind6;
   indices[6] = ind7;
   indices[7] = ind8;
}

void Hexahedron::GetVertices(Array<int> &v) const
{
   v.SetSize(8);
   for (int i = 0; i < 8; i++)
   {
      v[i] = indices[i];
   }
}

TriLinear3DFiniteElement HexahedronFE;

}
