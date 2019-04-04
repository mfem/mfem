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

//static method
void Pentatope::GetPointMatrix(unsigned transform, DenseMatrix &pm)
{
   double* a = &pm(0,0), *b = &pm(0,1), *c = &pm(0,2), *d = &pm(0,3), *e = &pm(0,
                                                                               4);

   // initialize to identity
   a[0] = 0.0, a[1] = 0.0, a[2] = 0.0, a[3] = 0.0;
   b[0] = 1.0, b[1] = 0.0, b[2] = 0.0, b[3] = 0.0;
   c[0] = 0.0, c[1] = 1.0, c[2] = 0.0, c[3] = 0.0;
   d[0] = 0.0, d[1] = 0.0, d[2] = 1.0, d[3] = 0.0;
   e[0] = 0.0, e[1] = 0.0, e[2] = 0.0, e[3] = 1.0;

#define ASGN(a, b) (a[0] = b[0], a[1] = b[1], a[2] = b[2], a[3] = b[3])
#define SWAP(a, b) for (int i = 0; i < 4; i++) { std::swap(a[i], b[i]); }
#define AVG(a, b, c) for (int i = 0; i < 4; i++) { a[i] = (b[i]+c[i])*0.5; }

   double f[4], g[4];
   switch (transform)
   {
      case 0 : AVG(b,a,b); AVG(c,a,c); AVG(d,a,d); AVG(e,a,e); break; // 1,6,7,8,9
      case 1 : AVG(a,a,b); AVG(c,b,c); AVG(d,b,d); AVG(e,b,e); break; // 6,2,10,11,12
      case 2 : AVG(a,a,c); AVG(b,b,c); AVG(d,c,d); AVG(e,c,e); break; // 7,10,3,13,14
      case 3 : AVG(a,a,d); AVG(b,b,d); AVG(c,c,d); AVG(e,d,e); break; // 8,11,13,4,15
      case 4 : AVG(a,a,e); AVG(b,b,e); AVG(c,c,e); AVG(d,d,e); break; // 9,12,14,15,5
      case 5 : ASGN(f,e); AVG(e,d,e); AVG(d,c,d); ASGN(g,b); AVG(c,b,c); AVG(b,a,f);
         AVG(a,a,g); break; // 6,9,10,13,15
      case 6 : ASGN(f,a); AVG(a,a,b); AVG(b,f,d); ASGN(g,c); AVG(c,f,e); AVG(e,d,e);
         AVG(d,g,d); break; // 6,8,9,13,15
      case 7 : ASGN(f,e); AVG(e,d,e); AVG(d,b,d); ASGN(g,c); AVG(c,b,c); AVG(b,a,f);
         AVG(a,a,g); break; // 7,9,10,11,13
      case 8 : ASGN(f,a); AVG(a,a,b); AVG(b,f,c); ASGN(g,e); AVG(e,c,d); AVG(c,f,d);
         AVG(d,f,g); break; // 6,7,8,9,13
      case 9 : ASGN(f,e); AVG(a,a,e); AVG(e,f,d); ASGN(g,c); AVG(b,b,c); AVG(c,c,d);
         AVG(d,f,g); break; // 9,10,13,14,15
      case 10: ASGN(f,a); AVG(a,a,b); AVG(c,b,c); ASGN(g,e); AVG(e,d,e); AVG(d,b,g);
         AVG(b,f,g); break; // 6,9,10,12,15
      case 11: ASGN(f,d); AVG(d,c,d); AVG(e,d,e); ASGN(g,a); AVG(a,a,b); AVG(c,b,f);
         AVG(b,g,f); break; // 6,8,11,13,15
      case 12: ASGN(f,b); AVG(a,a,b); AVG(b,b,c); ASGN(g,e); AVG(e,d,e); AVG(c,f,d);
         AVG(d,f,g); break; // 6,10,11,12,15
      case 13: ASGN(f,c); AVG(c,a,e); AVG(e,c,d); ASGN(g,a); AVG(a,a,b); AVG(d,b,f);
         AVG(b,f,g); break; // 6,7,9,10,13
      case 14: ASGN(f,b); AVG(b,b,c); AVG(a,a,e); ASGN(g,e); AVG(e,d,e); AVG(d,c,g);
         AVG(c,f,g); break; // 9,10,12,14,15
      case 15: ASGN(f,d); AVG(d,c,d); AVG(a,a,b); ASGN(g,b); AVG(b,b,c); AVG(e,f,e);
         AVG(c,g,f); break; // 6,10,11,13,15
      default:
         MFEM_ABORT("Invalid transform.");
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
