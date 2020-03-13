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

// Implementation of data types for sparse matrix smoothers

#include "vector.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "sparsesmoothers.hpp"
#include <iostream>

namespace mfem
{

void SparseSmoother::SetOperator(const Operator &a)
{
   oper = dynamic_cast<const SparseMatrix*>(&a);
   if (oper == NULL)
   {
      mfem_error("SparseSmoother::SetOperator : not a SparseMatrix!");
   }
   height = oper->Height();
   width = oper->Width();
}

/// Matrix vector multiplication with GS Smoother.
void GSSmoother::Mult(const Vector &x, Vector &y) const
{
   if (!iterative_mode)
   {
      y = 0.0;
   }
   for (int i = 0; i < iterations; i++)
   {
      if (type != 2)
      {
         oper->Gauss_Seidel_forw(x, y);
      }
      if (type != 1)
      {
         oper->Gauss_Seidel_back(x, y);
      }
   }
}

/// Create the Jacobi smoother.
DSmoother::DSmoother(const SparseMatrix &a, int t, double s, int it)
   : SparseSmoother(a)
{
   type = t;
   scale = s;
   iterations = it;
}

/// Matrix vector multiplication with Jacobi smoother.
void DSmoother::Mult(const Vector &x, Vector &y) const
{
   if (!iterative_mode && type == 0 && iterations == 1)
   {
      oper->DiagScale(x, y, scale);
      return;
   }

   z.SetSize(width);

   Vector *r = &y, *p = &z;

   if (iterations % 2 == 0)
   {
      Swap<Vector*>(r, p);
   }

   if (!iterative_mode)
   {
      *p = 0.0;
   }
   else if (iterations % 2)
   {
      *p = y;
   }
   for (int i = 0; i < iterations; i++)
   {
      if (type == 0)
      {
         oper->Jacobi(x, *p, *r, scale);
      }
      else if (type == 1)
      {
         oper->Jacobi2(x, *p, *r, scale);
      }
      else if (type == 2)
      {
         oper->Jacobi3(x, *p, *r, scale);
      }
      else
      {
         mfem_error("DSmoother::Mult wrong type");
      }
      Swap<Vector*>(r, p);
   }
}

}
