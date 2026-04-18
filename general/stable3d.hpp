// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

/** @brief Symmetric 3D Table stored as an array of rows each of which has a
    stack of column, floor, number nodes. The number of the node is assigned by
    counting the nodes from zero as they are pushed into the table. Diagonals of
    any kind are not allowed so the row, column and floor must all be different
    for each node. Only one node is stored for all 6 symmetric entries that are
    indexable by unique triplets of row, column, and floor. */
class STable3D
{
private:
   int Size, NElem;
   STable3DNode **Rows;

#ifdef MFEM_USE_MEMALLOC
   MemAlloc <STable3DNode, 1024> NodesMem;
#endif

public:
   /// Construct the table with a total of 'nr' rows.
   explicit STable3D (int nr);

   /** @brief Check to see if this entry is in the table and add it to the table
       if it is not there. Returns the number assigned to the table entry. */
   int Push (int r, int c, int f);

   /// Return the number assigned to the table entry. Abort if it's not there.
   int operator() (int r, int c, int f) const;

   /** Return the number assigned to the table entry. Return -1 if it's not
       there. */
   int Index (int r, int c, int f) const;

   /** @brief Check to see if this entry is in the table and add it to the table
       if it is not there. The entry is addressed by the three smallest values
       of (r,c,f,t). Returns the number assigned to the table entry. */
   int Push4 (int r, int c, int f, int t);

   /** @brief Return the number assigned to the table entry. The entry is
       addressed by the three smallest values of (r,c,f,t). Return -1 if it is
       not there. */
   int operator() (int r, int c, int f, int t) const;

   /// Return the number of elements added to the table.
   int NumberOfElements() { return NElem; }

   /// Print out all of the table elements.
   void Print(std::ostream &out = mfem::out) const;

   ~STable3D ();
};

}

#endif
