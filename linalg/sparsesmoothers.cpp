// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of data types for sparse matrix smoothers

#include <iostream>
#include "vector.hpp"
#include "matrix.hpp"
#include "sparsemat.hpp"
#include "sparsesmoothers.hpp"

/// Create GSSmoother.
GSSmoother::GSSmoother(const SparseMatrix & a) : MatrixInverse(a)
{}

/// Matrix vector multiplication with GS Smoother.
void GSSmoother::Mult(const Vector &x, Vector &y) const
{
   y = 0.;

   ((const SparseMatrix *)a)->Gauss_Seidel_forw(x, y);
   ((const SparseMatrix *)a)->Gauss_Seidel_back(x, y);
}

/// Destroys the GS Smoother.
GSSmoother::~GSSmoother()
{}

/// Create the diagonal smoother.
DSmoother::DSmoother(const SparseMatrix &a, double s) : MatrixInverse(a)
{
   scale = s;
}

/// Matrix vector multiplication with Diagonal smoother.
void DSmoother::Mult(const Vector &x, Vector &y) const
{
   for (int i = 0; i < x.Size(); i++)
      y(i) = scale * x(i) / a->Elem(i, i);
}

/// Destroys the Diagonal smoother.
DSmoother::~DSmoother()
{}
