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

#include <iostream>

namespace mfem
{

class STable3DNode
{
public:
   STable3DNode *Prev;
   int Column, Floor, Tier, Number;
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

   int Push (int r, int c, int f, int t = -1);

   int operator() (int r, int c, int f) const;

   int Index (int r, int c, int f) const;

   int Push4 (int r, int c, int f, int t);

   int operator() (int r, int c, int f, int t) const;

   int NumberOfRows() const { return Size; }
   int NumberOfElements() const { return NElem; }

   void Print(std::ostream &out = mfem::out) const;

   ~STable3D ();

   class RowIterator
   {
   private:
      STable3DNode *n;
   public:
      RowIterator (const STable3D &t, int r) { n = t.Rows[r]; }
      int operator!() { return (n != NULL); }
      void operator++() { n = n->Prev; }
      int Column() { return (n->Column); }
      int Floor()  { return (n->Floor); }
      int Tier()   { return (n->Tier); }
      int Index()  { return (n->Number); }
   };
};

}

#endif
