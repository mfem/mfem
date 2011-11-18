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

#ifndef MFEM_SPARSEMATSMOOTHERS
#define MFEM_SPARSEMATSMOOTHERS

/// Data type for symmetric Gauss-Seidel smoother of sparse matrix
class GSSmoother : public MatrixInverse
{
public:

   /// Create GSSmoother.
   GSSmoother(const SparseMatrix &a);

   /// Matrix vector multiplication with GS Smoother.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Destroys the GS Smoother.
   virtual ~GSSmoother();
};

/// Data type for scaled Diagonal smoother of sparse matrix
class DSmoother : public MatrixInverse
{
private:
   /// Scale for the Diagonal smooter.
   double scale;

public:

   /// Create the diagonal smoother.
   DSmoother(const SparseMatrix &a, double scale = 1.);

   /// Matrix vector multiplication with Diagonal smoother.
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Destroys the Diagonal smoother.
   virtual ~DSmoother();
};

#endif
