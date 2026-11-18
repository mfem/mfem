// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

#include <array>

namespace mfem
{

Triangle::Triangle(const int *ind, int attr) : Element(Geometry::TRIANGLE)
{
   attribute = attr;
   for (int i = 0; i < 3; i++)
   {
      indices[i] = ind[i];
   }
   transform = 0;
}

Triangle::Triangle(int ind1, int ind2, int ind3, int attr)
   : Element(Geometry::TRIANGLE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   transform = 0;
}

int Triangle::NeedRefinement(HashTable<Hashed2> &v_to_v) const
{
   if (v_to_v.FindId(indices[0], indices[1]) != -1) { return 1; }
   if (v_to_v.FindId(indices[1], indices[2]) != -1) { return 1; }
   if (v_to_v.FindId(indices[2], indices[0]) != -1) { return 1; }
   return 0;
}

void Triangle::SetVertices(const int *ind)
{
   for (int i = 0; i < 3; i++)
   {
      indices[i] = ind[i];
   }
}

// static method
void Triangle::MarkEdge(int *indices, const DSTable &v_to_v,
                        const Array<real_t> &length,
                        const Array<std::array<real_t, 6>> &keys)
{
   int e, j, ind[3];
   real_t l, L;
   std::array<real_t, 6> l2, L2;
   // The edges are ordered by their lengths, and edges of equal length by
   // keys: a strict total order (the keys of the edges of an element differ
   // unless two of its vertices coincide), so that the longest edge of a face
   // is the same whatever the order in which its edges are compared (here and
   // in Tetrahedron::MarkEdge()). A tolerance on the lengths would not be
   // transitive.
   auto Compare = [&length, &keys, &l, &l2, &L, &L2](int edge)
   {
      l = length[edge];
      l2 = keys[edge];
      MFEM_ASSERT(l2 != L2, "Two edges of an element have the same key: two of "
                  "its vertices coincide to working precision");
      return (l > L || (l == L && l2 > L2));
   };

   // Find j as the index of the longest edge, using the pair of metrics
   e = v_to_v(indices[0], indices[1]);
   L = length[e]; L2 = keys[e];
   j = 0;
   if (Compare(v_to_v(indices[1], indices[2]))) { L = l; L2 = l2; j = 1; }
   if (Compare(v_to_v(indices[2], indices[0]))) { j = 2; }

   for (int i = 0; i < 3; i++)
   {
      ind[i] = indices[i];
   }

   switch (j)
   {
      case 1:
         indices[0] = ind[1]; indices[1] = ind[2]; indices[2] = ind[0];
         break;
      case 2:
         indices[0] = ind[2]; indices[1] = ind[0]; indices[2] = ind[1];
         break;
   }
}

// static method
void Triangle::GetPointMatrix(unsigned transform, DenseMatrix &pm)
{
   real_t *a = &pm(0,0), *b = &pm(0,1), *c = &pm(0,2);

   // initialize to identity
   a[0] = 0.0; a[1] = 0.0;
   b[0] = 1.0; b[1] = 0.0;
   c[0] = 0.0; c[1] = 1.0;

   int chain[12], n = 0;
   while (transform)
   {
      chain[n++] = (transform & 7) - 1;
      transform >>= 3;
   }

   /* The transformations and orientations here match
      Mesh::UniformRefinement and Mesh::Bisection for triangles:

          c                      c
           *                      *
           | \                    |\\
           |   \                  | \ \
           |  2  \  e             |  \  \
         f *-------*              |   \   \
           | \   3 | \            |    \    \
           |   \   |   \          |  4  \  5  \
           |  0  \ |  1  \        |      \      \
           *-------*-------*      *-------*-------*
          a        d        b    a        d        b
   */

   real_t d[2], e[2], f[2];
#define ASGN(a, b) (a[0] = b[0], a[1] = b[1])
#define AVG(a, b, c) (a[0] = (b[0] + c[0])*0.5, a[1] = (b[1] + c[1])*0.5)

   while (n)
   {
      switch (chain[--n])
      {
         case 0: AVG(b, a, b); AVG(c, a, c); break;
         case 1: AVG(a, a, b); AVG(c, b, c); break;
         case 2: AVG(a, a, c); AVG(b, b, c); break;

         case 3:
            AVG(d, a, b); AVG(e, b, c); AVG(f, c, a);
            ASGN(a, e); ASGN(b, f); ASGN(c, d); break;

         case 4:
            AVG(d, a, b); // N.B.: orientation
            ASGN(b, a); ASGN(a, c); ASGN(c, d); break;

         case 5:
            AVG(d, a, b); // N.B.: orientation
            ASGN(a, b); ASGN(b, c); ASGN(c, d); break;

         default:
            MFEM_ABORT("Invalid transform.");
      }
   }
}

void Triangle::GetVertices(Array<int> &v) const
{
   v.SetSize(3);
   std::copy(indices, indices + 3, v.begin());
}

void Triangle::SetVertices(const Array<int> &v)
{
   MFEM_ASSERT(v.Size() == 3, "!");
   std::copy(v.begin(), v.end(), indices);
}

} // namespace mfem
