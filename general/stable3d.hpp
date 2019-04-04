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

#ifndef MFEM_STABLE3D
#define MFEM_STABLE3D

#include "mem_alloc.hpp"
#include "../general/globals.hpp"

namespace mfem
{

class STable3DNode
{
public:
   STable3DNode *Prev;
   int Column, Floor, Number;
};

/// Symmetric 3D Table
class STable3D
{
private:
   int Size, NElem;
   STable3DNode **Rows;

#ifdef MFEM_USE_MEMALLOC
   MemAlloc <STable3DNode, 1024> NodesMem;
#endif

public:
   explicit STable3D (int nr);

   int Push (int r, int c, int f);

   int operator() (int r, int c, int f) const;

   int Index (int r, int c, int f) const;

   int Push4 (int r, int c, int f, int t);

   int operator() (int r, int c, int f, int t) const;

   int NumberOfElements() { return NElem; }

   void Print(std::ostream &out = mfem::out) const;

   ~STable3D ();
};


class STable4DNode
{
public:
   STable4DNode *Prev;
   int Column, Floor, Trace, Number;
};


/// Symmetric 4D Table
class STable4D
{
private:
   int Size, NElem;
   STable4DNode **Rows;

#ifdef MFEM_USE_MEMALLOC
   MemAlloc <STable4DNode, 1024> NodesMem;
#endif

public:
   explicit STable4D (int nr);

   int Push (int r, int c, int f, int t);

   int operator() (int r, int c, int f, int t) const;

   int Index (int r, int c, int f, int t) const;

   int NumberOfElements() { return NElem; };

   ~STable4D ();
};



class STable5DNode
{
public:
   STable5DNode *Prev;
   int Column, Floor, Trace, Next, Number;
};

/// Symmetric 5D Table
class STable5D
{
private:
   int Size, NElem;
   STable5DNode **Rows;

#ifdef MFEM_USE_MEMALLOC
   MemAlloc <STable5DNode, 1024> NodesMem;
#endif

public:
   explicit STable5D (int nr);

   int Push (int r, int c, int f, int t, int u);

   int operator() (int r, int c, int f, int t, int u) const;

   int Index (int r, int c, int f, int t, int u) const;

   int Push8 (int u1, int u2, int u3, int u4, int u5, int u6, int u7, int u8);
   int operator() (int u1, int u2, int u3, int u4, int u5, int u6, int u7,
                   int u8) const;

   int NumberOfElements() { return NElem; };

   ~STable5D ();
};


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


inline void Sort4 (int &r, int &c, int &f, int &u)
{
   Sort3(c,f,u);

   int t;

   if (r > c)
   {
      if (r <= f) //(r, c, f, u) --> (c, r, f, u)
      {
         t = r; r = c; c = t;
      }
      else if (r <= u) //(r, c, f, u) --> (c, f, r, u)
      {
         t = r; r = c; c = t;
         t = c; c = f; f = t;
      }
      else if (r > u) //(r, c, f, u) --> (c, f, u, r)
      {
         t = r; r = c; c = t;
         t = c; c = f; f = t;
         t = f; f = u; u = t;
      }
   }
}

inline void Sort5 (int &r, int &c, int &f, int &u, int &v)
{
   Sort4(r,c,f,u);
   Sort4(c,f,u,v);

   if (r > c)
   {
      int t = r; r = c; c = t;
   }
}

//should be optimized
inline void Sort8 (int &u1, int &u2, int &u3, int &u4, int &u5, int &u6,
                   int &u7, int &u8)
{
   Sort5(u1,u2,u3,u4,u5);
   Sort5(u4,u5,u6,u7,u8);

   Sort5(u1,u2,u3,u4,u5);
   Sort5(u4,u5,u6,u7,u8);

   Sort5(u1,u2,u3,u4,u5);
   Sort5(u4,u5,u6,u7,u8);

   Sort5(u1,u2,u3,u4,u5);
   Sort5(u4,u5,u6,u7,u8);
}


}

#endif
