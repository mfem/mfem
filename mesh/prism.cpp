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

// Implementation of class Prism

#include "mesh_headers.hpp"

namespace mfem
{

Prism::Prism(const int *ind, int attr)
   : Element(Geometry::PRISM)
{
   attribute = attr;
   for (int i = 0; i < 6; i++)
   {
      indices[i] = ind[i];
   }
   // refinement_flag = 0;
   // transform = 0;
}

Prism::Prism(int ind1, int ind2, int ind3, int ind4, int ind5, int ind6,
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
   // refinement_flag = 0;
   // transform = 0;
}
/*
void Prism::ParseRefinementFlag(int refinement_edges[2], int &type,
            int &flag)
{
   int i, f = refinement_flag;

   MFEM_VERIFY(f != 0, "prism is not marked");

   for (i = 0; i < 2; i++)
   {
      refinement_edges[i] = f & 7;
      f = f >> 3;
   }
   type = f & 7;
   flag = (f >> 3);
}

void Prism::CreateRefinementFlag(int refinement_edges[2], int type,
                                       int flag)
{
   // Check for correct type
#ifdef MFEM_DEBUG
   int e1, e2;
   e1 = refinement_edges[0];
   e2 = refinement_edges[1];
   // if (e1 > e2)  e1 = e2, e2 = refinement_edges[0];
   switch (type)
   {
      case Prism::TYPE_PU:
         if (e1 == 2 && e2 == 1) { break; }
         // if (e1 == 3 && e2 == 4) break;
         mfem_error("Error in Prism::CreateRefinementFlag(...) #1");
         break;
      case Prism::TYPE_A:
         if (e1 == 3 && e2 == 1) { break; }
         if (e1 == 2 && e2 == 4) { break; }
         // if (flag == 0)  // flag is assumed to be the generation
         //   if (e2 == 5)
         //      if (e1 >= 1 && e1 <= 5) break; // type is actually O or M
         //                                     //   ==>  ok for generation = 0
         mfem_error("Error in Prism::CreateRefinementFlag(...) #2");
         break;
      case Prism::TYPE_PF:
         if (flag > 0)  // PF is ok only for generation > 0
         {
            if (e1 == 2 && e2 == 1) { break; }
            // if (e1 == 3 && e2 == 4) break;
         }
         mfem_error("Error in Prism::CreateRefinementFlag(...) #3");
         break;
      case Prism::TYPE_O:
         if (flag == 0 && e1 == 5 && e2 == 5)
         {
            break;
         }
         mfem_error("Error in Prism::CreateRefinementFlag(...) #4");
         break;
      case Prism::TYPE_M:
         if (flag == 0)
         {
            if (e1 == 5 && e2 == 1) { break; }
            if (e1 == 2 && e2 == 5) { break; }
         }
         mfem_error("Error in Prism::CreateRefinementFlag(...) #5");
         break;
      default:
         mfem_error("Error in Prism::CreateRefinementFlag(...) #6");
         break;
   }
#endif

   refinement_flag = flag;
   refinement_flag <<= 3;

   refinement_flag |= type;
   refinement_flag <<= 3;

   refinement_flag |= refinement_edges[1];
   refinement_flag <<= 3;

   refinement_flag |= refinement_edges[0];
}

void Prism::GetMarkedFace(const int face, int *fv)
{
   int re[2], type, flag, *tv = this->indices;
   ParseRefinementFlag(re, type, flag);
   switch (face)
   {
      case 0:
         switch (re[1])
         {
            case 1: fv[0] = tv[1]; fv[1] = tv[2]; fv[2] = tv[3]; break;
            case 4: fv[0] = tv[3]; fv[1] = tv[1]; fv[2] = tv[2]; break;
            case 5: fv[0] = tv[2]; fv[1] = tv[3]; fv[2] = tv[1]; break;
         }
         break;
      case 1:
         switch (re[0])
         {
            case 2: fv[0] = tv[2]; fv[1] = tv[0]; fv[2] = tv[3]; break;
            case 3: fv[0] = tv[0]; fv[1] = tv[3]; fv[2] = tv[2]; break;
            case 5: fv[0] = tv[3]; fv[1] = tv[2]; fv[2] = tv[0]; break;
         }
         break;
      case 2:
         fv[0] = tv[0]; fv[1] = tv[1]; fv[2] = tv[3];
         break;
      case 3:
         fv[0] = tv[1]; fv[1] = tv[0]; fv[2] = tv[2];
         break;
   }
}

int Prism::NeedRefinement(DSTable &v_to_v, int *middle) const
{
   int m;

   if ((m = v_to_v(indices[0], indices[1])) != -1)
      if (middle[m] != -1) { return 1; }
   if ((m = v_to_v(indices[1], indices[2])) != -1)
      if (middle[m] != -1) { return 1; }
   if ((m = v_to_v(indices[2], indices[0])) != -1)
      if (middle[m] != -1) { return 1; }
   if ((m = v_to_v(indices[0], indices[3])) != -1)
      if (middle[m] != -1) { return 1; }
   if ((m = v_to_v(indices[1], indices[3])) != -1)
      if (middle[m] != -1) { return 1; }
   if ((m = v_to_v(indices[2], indices[3])) != -1)
      if (middle[m] != -1) { return 1; }
   return 0;
}
*/
void Prism::SetVertices(const int *ind)
{
   for (int i = 0; i < 6; i++)
   {
      indices[i] = ind[i];
   }
}
/*
void Prism::MarkEdge(const DSTable &v_to_v, const int *length)
{
   int ind[4], i, j, l, L, type;

   // determine the longest edge
   L = length[v_to_v(indices[0], indices[1])]; j = 0;
   if ((l = length[v_to_v(indices[1], indices[2])]) > L) { L = l; j = 1; }
   if ((l = length[v_to_v(indices[2], indices[0])]) > L) { L = l; j = 2; }
   if ((l = length[v_to_v(indices[0], indices[3])]) > L) { L = l; j = 3; }
   if ((l = length[v_to_v(indices[1], indices[3])]) > L) { L = l; j = 4; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; j = 5; }

   for (i = 0; i < 4; i++)
   {
      ind[i] = indices[i];
   }

   switch (j)
   {
      case 1:
         indices[0] = ind[1]; indices[1] = ind[2];
         indices[2] = ind[0]; indices[3] = ind[3];
         break;
      case 2:
         indices[0] = ind[2]; indices[1] = ind[0];
         indices[2] = ind[1]; indices[3] = ind[3];
         break;
      case 3:
         indices[0] = ind[3]; indices[1] = ind[0];
         indices[2] = ind[2]; indices[3] = ind[1];
         break;
      case 4:
         indices[0] = ind[1]; indices[1] = ind[3];
         indices[2] = ind[2]; indices[3] = ind[0];
         break;
      case 5:
         indices[0] = ind[2]; indices[1] = ind[3];
         indices[2] = ind[0]; indices[3] = ind[1];
         break;
   }

   // Determine the two longest edges for the other two faces and
   // store them in ind[0] and ind[1]
   ind[0] = 2; ind[1] = 1;
   L = length[v_to_v(indices[0], indices[2])];
   if ((l = length[v_to_v(indices[0], indices[3])]) > L) { L = l; ind[0] = 3; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; ind[0] = 5; }

   L = length[v_to_v(indices[1], indices[2])];
   if ((l = length[v_to_v(indices[1], indices[3])]) > L) { L = l; ind[1] = 4; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; ind[1] = 5; }

   j = 0;
   switch (ind[0])
   {
      case 2:
         switch (ind[1])
         {
            case 1:  type = Prism::TYPE_PU; break;
            case 4:  type = Prism::TYPE_A;  break;
            case 5:
            default: type = Prism::TYPE_M;
         }
         break;
      case 3:
         switch (ind[1])
         {
            case 1:  type = Prism::TYPE_A;  break;
            case 4:  type = Prism::TYPE_PU;
               j = 1; ind[0] = 2; ind[1] = 1; break;
            case 5:
            default: type = Prism::TYPE_M;
               j = 1; ind[0] = 5; ind[1] = 1;
         }
         break;
      case 5:
      default:
         switch (ind[1])
         {
            case 1:  type = Prism::TYPE_M;  break;
            case 4:  type = Prism::TYPE_M;
               j = 1; ind[0] = 2; ind[1] = 5; break;
            case 5:
            default: type = Prism::TYPE_O;
         }
   }

   if (j)
   {
      j = indices[0]; indices[0] = indices[1]; indices[1] = j;
      j = indices[2]; indices[2] = indices[3]; indices[3] = j;
   }

   CreateRefinementFlag(ind, type);
}
*/
/*
// static method
void Prism::GetPointMatrix(unsigned transform, DenseMatrix &pm)
{
   double *a = &pm(0,0), *b = &pm(0,1), *c = &pm(0,2), *d = &pm(0,3);

   // initialize to identity
   a[0] = 0.0, a[1] = 0.0, a[2] = 0.0;
   b[0] = 1.0, b[1] = 0.0, b[2] = 0.0;
   c[0] = 0.0, c[1] = 1.0, c[2] = 0.0;
   d[0] = 0.0, d[1] = 0.0, d[2] = 1.0;

   int chain[12], n = 0;
   while (transform)
   {
      chain[n++] = (transform & 7) - 1;
      transform >>= 3;
   }

   // The transformations and orientations here match the six cases in
   // Mesh::Bisection for tetrahedra.
   while (n)
   {
#define ASGN(a, b) (a[0] = b[0], a[1] = b[1], a[2] = b[2])
#define SWAP(a, b) for (int i = 0; i < 3; i++) { std::swap(a[i], b[i]); }
#define AVG(a, b, c) for (int i = 0; i < 3; i++) { a[i] = (b[i]+c[i])*0.5; }

      double e[3];
      AVG(e, a, b);
      switch (chain[--n])
      {
         case 0: ASGN(b, c); ASGN(c, d); break;
         case 1: ASGN(a, c); ASGN(c, d); break;
         case 2: ASGN(b, a); ASGN(a, d); break;
         case 3: ASGN(a, b); ASGN(b, d); break;
         case 4: SWAP(a, c); ASGN(b, d); break;
         case 5: SWAP(b, c); ASGN(a, d); break;
         default:
            MFEM_ABORT("Invalid transform.");
      }
      ASGN(d, e);
   }
}
*/
void Prism::GetVertices(Array<int> &v) const
{
   v.SetSize(6);
   for (int i = 0; i < 6; i++)
   {
      v[i] = indices[i];
   }
}
/*
Element *Prism::Duplicate(Mesh *m) const
{
#ifdef MFEM_USE_MEMALLOC
   Prism *pri = m->PriMemory.Alloc();
#else
   Prism *pri = new Prism;
#endif
   pri->SetVertices(indices);
   pri->SetAttribute(attribute);
   pri->SetRefinementFlag(refinement_flag);
   return pri;
}
*/
BiLinear3DFiniteElement PrismFE;

}
