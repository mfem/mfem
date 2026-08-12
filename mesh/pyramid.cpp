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

// Implementation of class Pyramid

#include "mesh_headers.hpp"

namespace mfem
{

Pyramid::Pyramid(const int *ind, int attr)
   : Element(Geometry::PYRAMID)
{
   attribute = attr;
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }
}

Pyramid::Pyramid(int ind1, int ind2, int ind3, int ind4, int ind5, int attr)
   : Element(Geometry::PYRAMID)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;
}

void Pyramid::SetVertices(const int *ind)
{
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }
}

void Pyramid::GetVertices(Array<int> &v) const
{
   v.SetSize(5);
   std::copy(indices, indices + 5, v.begin());
}

void Pyramid::SetVertices(const Array<int> &v)
{
   MFEM_ASSERT(v.Size() == 5, "!");
   std::copy(v.begin(), v.end(), indices);
}

int Pyramid::GetNFaces(int &nFaceVertices) const
{
   MFEM_ABORT("this method is not valid for Pyramid elements");
   nFaceVertices = 4;
   return 5;
}

}
