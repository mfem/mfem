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
