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
         dinv(i*m+j) = 1./(a(i) + b(j));  
}

void KronMultInvDiag(const Vector & a, const Vector & b, 
                     const Vector & c, Vector & dinv)
{
   int n = a.Size(), m = b.Size(), l = c.Size();
   dinv.SetSize(n*m*l);

   for (int k = 0; k<l; k++)
      for (int j = 0; j<m; j++)
         for (int i = 0; i<n; i++)
            dinv(i*m*l+j*l+k) = 1.0/(a(i) + b(j) + c(k));
}

FDSolver::FDSolver(Array<DenseMatrix *> A_, Array<DenseMatrix *> B_)
: A(A_), B(B_)
{
   if (A.Size() && B.Size())
      MFEM_VERIFY(A.Size() == B.Size(), "FDSolver: Incompatible Dimensions");
   
   size = A[0]->Height();
   if (size) Setup();
}

void FDSolver::Setup()
{
   dim = B.Size();
   Sd.SetSize(dim);
   Array<DenseMatrix * > D(dim); // D[i] = B[i]^-1 * A[i]
   for (int i=0; i<dim; i++)
   {
      Sd[i] = nullptr;
      int j = dim-i-1;
      if (B[j]->Height()) 
      {
         Sd[i] = new DenseMatrixInverse(*B[j]);   
      }
   }
   if (A.Size())
   {
      SQ.SetSize(dim);
      SQinv.SetSize(dim);
      for (int i=0; i<dim; i++)
      {
         int j = dim-i-1;
         int n = A[j]->Height();
         D[i] = new DenseMatrix(n);
         if(n)
         {
            // B_i^-1 * A_i^-1
            Sd[i]->Mult(*A[j],*D[i]);
         }
      }
      Q.SetSize(dim);
      Diag.SetSize(dim);
      int size = 1;
      for (int i = 0; i<dim; i++)
      {
         SQ[i] = nullptr;
         size *= D[i]->Height();
         // Diag[i] = new Vector;
         // Q[i] = new DenseMatrix;
         if (D[i]->Height())
         {
            // D[i]->Eigensystem(*Diag[i],*Q[i]); // non-spd
            DenseMatrixEigensystem Eig(*D[i]);
            Diag[i] = new Vector(Eig.Eigenvalues()); // non-spd
            Q[i] = new DenseMatrix(Eig.Eigenvectors()); // non-spd
            DenseMatrixInverse Qinv(*Q[i]);
            DenseMatrix Sdinv;
            Sd[i]->GetInverseMatrix(Sdinv);
            SQ[i] = new DenseMatrix;
            Qinv.Mult(Sdinv,*SQ[i]);
         }
      }
      if (dim == 2)
      {
         KronMultInvDiag(*Diag[0], *Diag[1], dinv);
      }
      else
      {
         KronMultInvDiag(*Diag[0], *Diag[1], *Diag[2], dinv);
      }
      for (int i=0; i<dim; i++) { delete D[i]; };
   }
}


} // namespace mfem