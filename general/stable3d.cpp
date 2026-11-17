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


#include "error.hpp"
#include "stable3d.hpp"

#include <algorithm>

using namespace std;

namespace mfem
{

STable3D::STable3D (int nr)
{
   int i;

   Size = nr;
   Rows = new STable3DNode *[nr];
   for (i = 0; i < nr; i++)
   {
      Rows[i] = NULL;
   }
   NElem = 0;
}

inline void Sort3 (int &r, int &c, int &f)
{
   int t;

   if (r > c)
      if (c > f)
      {
         t = r;  r = f;  f = t;  //  (r,c,f) --> (f,c,r)
      }
      else if (r > f)
      {
         t = r;  r = c;  c = f;  f = t;  //  (r,c,f) --> (c,f,r)
      }
      else
      {
         t = r;  r = c;  c = t;  //  (r,c,f) --> (c,r,f)
      }
   else if (c > f)
   {
      if (r > f)
      {
         t = f;  f = c;  c = r;  r = t;  //  (r,c,f) --> (f,r,c)
      }
      else
      {
         t = c;  c = f;  f = t;  //  (r,c,f) --> (r,f,c)
      }
   }
}

int STable3D::Push (int r, int c, int f)
{
   STable3DNode *node;

   MFEM_ASSERT(r != c && c != f && f != r,
               "STable3D::Push : r = " << r << ", c = " << c << ", f = " << f);

   Sort3 (r, c, f);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f && node->Fourth == -1)
         {
            return node->Number;
         }
   }

#ifdef MFEM_USE_MEMALLOC
   node = NodesMem.Alloc ();
#else
   node = new STable3DNode;
#endif
   node->Column = c;
   node->Floor  = f;
   node->Fourth = -1;
   node->Number = NElem;
   node->Prev   = Rows[r];
   Rows[r] = node;

   NElem++;
   return (NElem-1);
}

int STable3D::operator() (int r, int c, int f) const
{
   STable3DNode *node;

   Sort3 (r, c, f);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f && node->Fourth == -1)
         {
            return node->Number;
         }
   }

   MFEM_ABORT("(r,c,f) = (" << r << "," << c << "," << f << ")");

   return 0;
}

int STable3D::Index (int r, int c, int f) const
{
   STable3DNode *node;

   if (r >= Size)
   {
      return -1;
   }

   Sort3 (r, c, f);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f && node->Fourth == -1)
         {
            return node->Number;
         }
   }

   return -1;
}

int STable3D::Push4 (int r, int c, int f, int t)
{
   MFEM_ASSERT(r != c && r != f && r != t && c != f && c != t && f != t,
               " r = " << r << ", c = " << c << ", f = " << f << ", t = " << t);
   int vertices[4] = {r, c, f, t};
   std::sort(vertices, vertices + 4);
   STable3DNode *node;
   for (node = Rows[vertices[0]]; node != NULL; node = node->Prev)
   {
      if (node->Column == vertices[1] && node->Floor == vertices[2] &&
          node->Fourth == vertices[3]) { return node->Number; }
   }
#ifdef MFEM_USE_MEMALLOC
   node = NodesMem.Alloc();
#else
   node = new STable3DNode;
#endif
   node->Column = vertices[1];
   node->Floor = vertices[2];
   node->Fourth = vertices[3];
   node->Number = NElem++;
   node->Prev = Rows[vertices[0]];
   Rows[vertices[0]] = node;
   return node->Number;
}

int STable3D::operator() (int r, int c, int f, int t) const
{
   const int index = Index(r, c, f, t);
   MFEM_VERIFY(index >= 0,
               "(r,c,f,t) = (" << r << "," << c << "," << f << "," << t << ")");
   return index;
}

int STable3D::Index (int r, int c, int f, int t) const
{
   int vertices[4] = {r, c, f, t};
   std::sort(vertices, vertices + 4);
   if (vertices[0] >= Size) { return -1; }
   for (STable3DNode *node = Rows[vertices[0]];
        node != NULL; node = node->Prev)
   {
      if (node->Column == vertices[1] && node->Floor == vertices[2] &&
          node->Fourth == vertices[3]) { return node->Number; }
   }
   return -1;
}

STable3D::~STable3D ()
{
#ifdef MFEM_USE_MEMALLOC
   // NodesMem.Clear();  // this is done implicitly
#else
   for (int i = 0; i < Size; i++)
   {
      STable3DNode *aux, *node_p = Rows[i];
      while (node_p != NULL)
      {
         aux = node_p;
         node_p = node_p->Prev;
         delete aux;
      }
   }
#endif
   delete [] Rows;
}

void STable3D::Print(std::ostream & os) const
{
   os << NElem << endl;
   for (int row = 0; row < Size; row++)
   {
      STable3DNode *node_p = Rows[row];
      while (node_p != NULL)
      {
         os << row
            << ' ' << node_p->Column
            << ' ' << node_p->Floor
            << ' ' << node_p->Fourth
            << ' ' << node_p->Number
            << endl;
         node_p = node_p->Prev;
      }
   }
}

}
