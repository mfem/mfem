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

// Implementation of class Tetrahedron

#include "mesh_headers.hpp"

namespace mfem
{

const int Tetrahedron::edges[6][2] =
{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

void Tetrahedron::ParseRefinementFlag(int refinement_edges[2], int &type,
                                      int &flag)
{
   int i, f = refinement_flag;

   if (f == 0)
      mfem_error("Tetrahedron::ParseRefinementFlag :"
                 " tetrahedron is not marked");

   for (i = 0; i < 2; i++)
   {
      refinement_edges[i] = f & 7;
      f = f >> 3;
   }
   type = f & 7;
   flag = (f >> 3);
}

void Tetrahedron::CreateRefinementFlag(int refinement_edges[2], int type,
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
      case Tetrahedron::TYPE_PU:
         if (e1 == 2 && e2 == 1) { break; }
         // if (e1 == 3 && e2 == 4) break;
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #1");
         break;
      case Tetrahedron::TYPE_A:
         if (e1 == 3 && e2 == 1) { break; }
         if (e1 == 2 && e2 == 4) { break; }
         // if (flag == 0)  // flag is assumed to be the generation
         //   if (e2 == 5)
         //      if (e1 >= 1 && e1 <= 5) break; // type is actually O or M
         //                                     //   ==>  ok for generation = 0
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #2");
         break;
      case Tetrahedron::TYPE_PF:
         if (flag > 0)  // PF is ok only for generation > 0
         {
            if (e1 == 2 && e2 == 1) { break; }
            // if (e1 == 3 && e2 == 4) break;
         }
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #3");
         break;
      case Tetrahedron::TYPE_O:
         if (flag == 0 && e1 == 5 && e2 == 5)
         {
            break;
         }
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #4");
         break;
      case Tetrahedron::TYPE_M:
         if (flag == 0)
         {
            if (e1 == 5 && e2 == 1) { break; }
            if (e1 == 2 && e2 == 5) { break; }
         }
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #5");
         break;
      default:
         mfem_error("Error in Tetrahedron::CreateRefinementFlag(...) #6");
         break;
   }
#endif

   refinement_flag = flag;
   refinement_flag = refinement_flag << 3;

   refinement_flag = refinement_flag | type;
   refinement_flag = refinement_flag << 3;

   refinement_flag = refinement_flag | refinement_edges[1];
   refinement_flag = refinement_flag << 3;

   refinement_flag = refinement_flag | refinement_edges[0];
}

Tetrahedron::Tetrahedron(const int *ind, int attr)
   : Element(Geometry::TETRAHEDRON)
{
   attribute = attr;
   for (int i = 0; i < 4; i++)
   {
      indices[i] = ind[i];
   }
   refinement_flag = 0;
}

Tetrahedron::Tetrahedron(int ind1, int ind2, int ind3, int ind4, int attr)
   : Element(Geometry::TETRAHEDRON)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   refinement_flag = 0;
}

int Tetrahedron::NeedRefinement(DSTable &v_to_v, int *middle) const
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

void Tetrahedron::SetVertices(const int *ind)
{
   for (int i = 0; i < 4; i++)
   {
      indices[i] = ind[i];
   }
}

void Tetrahedron::MarkEdge(const DSTable &v_to_v, const int *length)
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
            case 1:  type = Tetrahedron::TYPE_PU; break;
            case 4:  type = Tetrahedron::TYPE_A;  break;
            case 5:
            default: type = Tetrahedron::TYPE_M;
         }
         break;
      case 3:
         switch (ind[1])
         {
            case 1:  type = Tetrahedron::TYPE_A;  break;
            case 4:  type = Tetrahedron::TYPE_PU;
               j = 1; ind[0] = 2; ind[1] = 1; break;
            case 5:
            default: type = Tetrahedron::TYPE_M;
               j = 1; ind[0] = 5; ind[1] = 1;
         }
         break;
      case 5:
      default:
         switch (ind[1])
         {
            case 1:  type = Tetrahedron::TYPE_M;  break;
            case 4:  type = Tetrahedron::TYPE_M;
               j = 1; ind[0] = 2; ind[1] = 5; break;
            case 5:
            default: type = Tetrahedron::TYPE_O;
         }
   }

   if (j)
   {
      j = indices[0]; indices[0] = indices[1]; indices[1] = j;
      j = indices[2]; indices[2] = indices[3]; indices[3] = j;
   }

   CreateRefinementFlag(ind, type);
}

void Tetrahedron::GetVertices(Array<int> &v) const
{
   v.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      v[i] = indices[i];
   }
}

Element *Tetrahedron::Duplicate(Mesh *m) const
{
#ifdef MFEM_USE_MEMALLOC
   Tetrahedron *tet = m->TetMemory.Alloc();
#else
   Tetrahedron *tet = new Tetrahedron;
#endif
   tet->SetVertices(indices);
   tet->SetAttribute(attribute);
   tet->SetRefinementFlag(refinement_flag);
   return tet;
}

Linear3DFiniteElement TetrahedronFE;

}
