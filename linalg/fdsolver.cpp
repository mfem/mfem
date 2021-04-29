// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of sparse matrix

#include "linalg.hpp"

// #include <iostream>
// #include <iomanip>
// #include <cmath>
// #include <algorithm>
// #include <limits>
// #include <cstring>

namespace mfem
{

void KronMultInvDiag(const Vector & a, const Vector & b, Vector & dinv)
{
   int n = a.Size(), m = b.Size();
   dinv.SetSize(n*m);

   for (int j = 0; j<m; j++)
      for (int i = 0; i<n; i++)
      {
         dinv(i*m+j) = 1./(a(i) + b(j));
      }
}

void KronMultInvDiag(const Vector & a, const Vector & b,
                     const Vector & c, Vector & dinv)
{
   int n = a.Size(), m = b.Size(), l = c.Size();
   dinv.SetSize(n*m*l);

   for (int k = 0; k<l; k++)
      for (int j = 0; j<m; j++)
         for (int i = 0; i<n; i++)
         {
            dinv(i*m*l+j*l+k) = 1.0/(a(i) + b(j) + c(k));
         }
}

void KronMultInvDiag(const Array<Vector *> & X, Vector & diag)
{
   int dim = X.Size();
   if (dim == 2)
   {
      KronMultInvDiag(*X[0], *X[1], diag);
   }
   else if (dim == 3)
   {
      KronMultInvDiag(*X[0], *X[1], *X[2], diag);
   }
   else
   {
      MFEM_ABORT("KronMultInvDiag::Wrong dimension");
   }
}

FDSolver::FDSolver(const Array<DenseMatrix *> & A,
                   const Array<DenseMatrix *> & B)
{
   MFEM_ASSERT(A.Size() == B.Size(), "DenseFDSolver: Incompatible Dimensions");
   dim = A.Size();

   int solver_size = 1;
   for (int i = 0; i<dim; i++)
   {
      MFEM_ASSERT(A[i]->Height() == A[i]->Width(),
                  "DenseFDSolver: Matrix is not square");
      MFEM_ASSERT(B[i]->Height() == B[i]->Width(),
                  "DenseFDSolver: Matrix is not square");
      MFEM_ASSERT(A[i]->Height() == B[i]->Height(),
                  "DenseFDSolver: Matrices A and B have incompatible size");
      solver_size *= A[i]->Height();
   }
   this->height = solver_size;
   this->width  = solver_size;
   if (solver_size) { Setup(A,B); }
}

void FDSolver::Setup(const Array<DenseMatrix *> & A,
                     const Array<DenseMatrix *> & B)
{
   EigSystem.SetSize(dim);
   eigv.SetSize(dim);
   Array<Vector *> evalues(dim);
   SQ.SetSize(dim);
   DenseMatrix D;
   for (int i = 0; i<dim; i++)
   {
      DenseMatrixInverse Minv(*B[i]);
      Minv.Mult(*A[i],D);
      EigSystem[i] = new DenseMatrixEigensystem(D);
      EigSystem[i]->Eval();
      evalues[i] = &EigSystem[i]->Eigenvalues();
      eigv[i] = &EigSystem[i]->Eigenvectors();
      DenseMatrixInverse Qinv(*eigv[i]);
      DenseMatrix Sdinv;
      Minv.GetInverseMatrix(Sdinv);
      SQ[i] = new DenseMatrix;
      Qinv.Mult(Sdinv,*SQ[i]);
   }
   KronMultInvDiag(evalues,dinv);
}


void FDSolver::Mult(const Vector & r,Vector & z) const
{
   MFEM_ASSERT(height == r.Size(),
               "DenseFDSolver::Mult: Inconsistent vector size");
   if (r.Size() == 0) { return; }
   Vector rtemp;
   KronMult(SQ,r,rtemp);
   // 2. Diagonal solve;
   rtemp *= dinv;
   // 3. Modify RHS; z <-- (Q1 x Q2) rtemp
   KronMult(eigv,rtemp,z);
}

FDSolver::~FDSolver()
{
   if (height)
   {
      for (int i=0; i<dim; i++)
      {
         delete SQ[i];
         delete EigSystem[i];
      }
   }
}


} // namespace mfem