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

#ifndef MFEM_EIGENSOLVERS
#define MFEM_EIGENSOLVERS

#include "vector.hpp"
#include "operator.hpp"

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