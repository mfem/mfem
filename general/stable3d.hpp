// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

}

#endif
