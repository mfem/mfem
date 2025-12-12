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
   MFEM_VERIFY(oper != nullptr, "Operator must be a SparseMatrix");
   height = oper->Height();
   width = oper->Width();

   At.reset();
   oper_T = nullptr;
}

void SparseSmoother::EnsureTranspose() const
{
   if (oper_T) { return; }

   const real_t tol = 1e-14;
   if (oper->IsSymmetric() > tol * oper->MaxNorm())
   {
      At.reset(Transpose(*oper));
      oper_T = At.get();
   }
   else
   {
      At.reset();
      oper_T = oper;
   }
}

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

void GSSmoother::MultTranspose(const Vector &x, Vector &y) const
{
   EnsureTranspose();

   if (!iterative_mode)
   {
      y = 0.0;
   }

   for (int i = 0; i < iterations; i++)
   {
      if (type != 1)
      {
         oper_T->Gauss_Seidel_forw(x, y);
      }
      if (type != 2)
      {
         oper_T->Gauss_Seidel_back(x, y);
      }
   }
}

void DSmoother::Mult_(const SparseMatrix &A, const Vector &x, Vector &y) const
{
   if (!iterative_mode && type == 0 && iterations == 1)
   {
      A.DiagScale(x, y, scale, use_abs_diag);
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
         A.Jacobi(x, *p, *r, scale, use_abs_diag);
      }
      else if (type == 1)
      {
         A.Jacobi2(x, *p, *r, scale);
      }
      else if (type == 2)
      {
         A.Jacobi3(x, *p, *r, scale);
      }
      else
      {
         MFEM_ABORT("Invalid type.");
      }
      Swap<Vector*>(r, p);
   }
}

void DSmoother::Mult(const Vector &x, Vector &y) const
{
   Mult_(*oper, x, y);
}

void DSmoother::MultTranspose(const Vector &x, Vector &y) const
{
   if (iterations == 1 && !iterative_mode)
   {
      Mult_(*oper, x, y);
      return;
   }

   EnsureTranspose();
   MFEM_VERIFY(type == 0 || !At, "l1 or lumped Jacobi transpose not implemented"
               " for non-symmetric matrices");
   Mult_(*oper_T, x, y);
}

}
