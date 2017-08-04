// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of class Pentatope


#include "mesh_headers.hpp"

namespace mfem
{


Pentatope::Pentatope(const int *ind, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute = attr;
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }

   transform = 0;
}

Pentatope::Pentatope(int ind1, int ind2, int ind3, int ind4, int ind5, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;

   transform = 0;
}

void Pentatope::GetVertices(Array<int> &v) const
{
   v.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      v[i] = indices[i];
   }
}

void Pentatope::SetVertices(const int *ind)
{
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }
}

Element *Pentatope::Duplicate(Mesh *m) const
{
   Pentatope *pent = new Pentatope;
   pent->SetVertices(indices);
   pent->SetAttribute(attribute);
   return pent;
}

Linear4DFiniteElement PentatopeFE;

}
