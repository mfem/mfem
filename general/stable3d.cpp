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


#include <iostream>
#include "error.hpp"
#include "stable3d.hpp"

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

int STable3D::Push (int r, int c, int f)
{
   STable3DNode *node;

   MFEM_ASSERT(r != c && c != f && f != r,
               "STable3D::Push : r = " << r << ", c = " << c << ", f = " << f);

   Sort3 (r, c, f);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
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
         if (node->Floor == f)
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

   Sort3 (r, c, f);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
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

   int i = 0;
   int max = r;

   if (max < c) { max = c, i = 1; }
   if (max < f) { max = f, i = 2; }
   if (max < t) { i = 3; }

   switch (i)
   {
      case 0:
         return Push (c,f,t);
      case 1:
         return Push (r,f,t);
      case 2:
         return Push (r,c,t);
      case 3:
         return Push (r,c,f);
   }

   return -1;
}

int STable3D::operator() (int r, int c, int f, int t) const
{
   int i = 0;
   int max = r;

   if (max < c) { max = c, i = 1; }
   if (max < f) { max = f, i = 2; }
   if (max < t) { i = 3; }

   switch (i)
   {
      case 0:
         return (*this)(c,f,t);
      case 1:
         return (*this)(r,f,t);
      case 2:
         return (*this)(r,c,t);
      case 3:
         return (*this)(r,c,f);
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

void STable3D::Print(std::ostream & out) const
{
   out << NElem << endl;
   for (int row = 0; row < Size; row++)
   {
      STable3DNode *node_p = Rows[row];
      while (node_p != NULL)
      {
         out << row
             << ' ' << node_p->Column
             << ' ' << node_p->Floor
             << ' ' << node_p->Number
             << endl;
         node_p = node_p->Prev;
      }
   }
}

STable4D::STable4D (int nr)
{
   int i;

   Size = nr;
   Rows = new STable4DNode *[nr];
   for (i = 0; i < nr; i++)
   {
      Rows[i] = NULL;
   }
   NElem = 0;
}


int STable4D::Push (int r, int c, int f, int t)
{
   STable4DNode *node;

   MFEM_ASSERT(r != c && c != f && f != r && r!=t && c!=t && f!=t,
               "STable4D::Push : r = " << r << ", c = " << c << ", f = " << f << ", t = " <<
               t);

   Sort4(r, c, f, t);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
            {
               return node->Number;
            }
   }

#ifdef MFEM_USE_MEMALLOC
   node = NodesMem.Alloc ();
#else
   node = new STable4DNode;
#endif
   node->Column = c;
   node->Floor  = f;
   node->Trace = t;
   node->Number = NElem;
   node->Prev   = Rows[r];
   Rows[r] = node;

   NElem++;
   return (NElem-1);
}

int STable4D::operator() (int r, int c, int f, int t) const
{
   STable4DNode *node;

   Sort4(r, c, f, t);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
            {
               return node->Number;
            }
   }

   MFEM_ABORT("STable4D::operator(): (r,c,f,t) = (" << r << "," << c << "," << f <<
              "," << t <<")");

   return -1;
}

int STable4D::Index (int r, int c, int f, int t) const
{
   STable4DNode *node;

   Sort4(r, c, f, t);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
            {
               return node->Number;
            }
   }

   return -1;
}


STable4D::~STable4D ()
{
#ifdef MFEM_USE_MEMALLOC
   // NodesMem.Clear();  // this is done implicitly
#else
   for (int i = 0; i < Size; i++)
   {
      STable4DNode *aux, *node_p = Rows[i];
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




STable5D::STable5D (int nr)
{
   int i;

   Size = nr;
   Rows = new STable5DNode *[nr];
   for (i = 0; i < nr; i++)
   {
      Rows[i] = NULL;
   }
   NElem = 0;
}


int STable5D::Push (int r, int c, int f, int t, int u)
{
   STable5DNode *node;

   MFEM_ASSERT(r != c && c != f && f != r && r!=t && c!=t && f!=t && r!=u &&
               c!=u && f!=u && t!=u,
               "STable5D::Push : r = " << r << ", c = " << c << ", f = " << f << ", t = " << t
               << ", u = " << u);

   Sort5(r, c, f, t, u);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
               if (node->Next == u)
               {
                  return node->Number;
               }
   }

#ifdef MFEM_USE_MEMALLOC
   node = NodesMem.Alloc ();
#else
   node = new STable5DNode;
#endif
   node->Column = c;
   node->Floor  = f;
   node->Trace = t;
   node->Next = u;
   node->Number = NElem;
   node->Prev   = Rows[r];
   Rows[r] = node;

   NElem++;
   return (NElem-1);
}

int STable5D::operator() (int r, int c, int f, int t, int u) const
{
   STable5DNode *node;

   Sort5(r, c, f, t, u);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
               if (node->Next == u)
               {
                  return node->Number;
               }
   }

   MFEM_ABORT("STable4D::operator(): (r,c,f,t,u) = (" << r << "," << c << "," << f
              << "," << t << "," << u <<")");

   return 0;
}

int STable5D::Index (int r, int c, int f, int t, int u) const
{
   STable5DNode *node;

   Sort5(r, c, f, t, u);

   for (node = Rows[r]; node != NULL; node = node->Prev)
   {
      if (node->Column == c)
         if (node->Floor == f)
            if (node->Trace == t)
            {
               return node->Number;
            }
   }

   return -1;
}

int STable5D::Push8 (int u1, int u2, int u3, int u4, int u5, int u6, int u7,
                     int u8)
{
   Sort8(u1, u2, u3, u4, u5, u6, u7, u8);

   return (*this).Push(u1,u2,u3,u4,u5);
}

int STable5D::operator() (int u1, int u2, int u3, int u4, int u5, int u6,
                          int u7, int u8) const
{
   Sort8(u1, u2, u3, u4, u5, u6, u7, u8);

   return (*this)(u1,u2,u3,u4,u5);
}


STable5D::~STable5D ()
{
#ifdef MFEM_USE_MEMALLOC
   // NodesMem.Clear();  // this is done implicitly
#else
   for (int i = 0; i < Size; i++)
   {
      STable5DNode *aux, *node_p = Rows[i];
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


}
