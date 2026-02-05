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

// Implementation of class Wedge

#include "mesh_headers.hpp"

namespace mfem
{

Wedge::Wedge(const int *ind, int attr)
   : Element(Geometry::PRISM)
{
   attribute = attr;
   for (int i = 0; i < 6; i++)
   {
      indices[i] = ind[i];
   }
}

Wedge::Wedge(int ind1, int ind2, int ind3, int ind4, int ind5, int ind6,
             int attr)
   : Element(Geometry::PRISM)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;
   indices[5] = ind6;
}

void Wedge::SetVertices(const int *ind)
{
   for (int i = 0; i < 6; i++)
   {
      indices[i] = ind[i];
   }
}

void Wedge::GetVertices(Array<int> &v) const
{
   v.SetSize(6);
   std::copy(indices, indices + 6, v.begin());
}

void Wedge::SetVertices(const Array<int> &v)
{
   MFEM_ASSERT(v.Size() == 6, "!");
   std::copy(v.begin(), v.end(), indices);
}

int Wedge::GetNFaces(int &nFaceVertices) const
{
   MFEM_ABORT("this method is not valid for Wedge elements");
   nFaceVertices = 4;
   return 5;
}

}
