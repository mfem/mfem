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

#ifndef MFEM_EIGENSOLVERS
#define MFEM_EIGENSOLVERS

#include "vector.hpp"

namespace mfem
{

/// Abstract Eigensolver
class Eigensolver
{
public:

   Eigensolver();
   virtual ~Eigensolver() {}

   virtual void SetTol(double tol) = 0;
   virtual void SetMaxIter(int max_iter) = 0;
   virtual void SetPrintLevel(int logging) = 0;
   virtual void SetNumModes(int num_eigs) = 0;

   virtual void SetOperator(Operator & A) = 0;
   virtual void SetMassMatrix(Operator & M) = 0;

   /// Perform the eigenvalue solve
   virtual void Solve() = 0;

   /// Collect the converged eigenvalues
   virtual void GetEigenvalues(Array<double> & eigenvalues) = 0;

   /// Extract a single eigenvector
   virtual Vector & GetEigenvector(unsigned int i) = 0;

   /// Transfer ownership of the converged eigenvectors
   virtual Vector ** StealEigenvectors() = 0;

};

}

#endif
