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

#ifndef MFEM_FDSOLVER
#define MFEM_FDSOLVER

#include "../config/config.hpp"
#include "densemat.hpp"

namespace mfem
{

/// Computes the inverse diagonal dinv = (a⊗I + I⊗b)^-1
/// where a, b are diagonal matrices and I is the identity of the
/// appropriate size
void KronProdInvDiag(const Vector & a, const Vector & b, Vector & dinv);

/// Computes the inverse diagonal dinv = (a⊗I⊗I + I⊗b⊗I + I⊗I⊗c)^-1
/// where a, b, c are diagonal matrices and I is the identity of the
/// appropriate size
void KronProdInvDiag(const Vector & a, const Vector & b,
                     const Vector & c, Vector & dinv);

void KronProdInvDiag(const Array<Vector *> & X, Vector & dinv);

#ifdef MFEM_USE_LAPACK

/// In 2D it solves the system (A_0 ⊗ B_1 + B_0 ⊗ A_1) z = r
/// In 3D it solves the system
///     (A_0 ⊗ B_1 ⊗ B_2 + B_0 ⊗ A_1 ⊗ B_2 + B_0 ⊗ B_1 ⊗ A_2) z = r
class FDSolver: public Solver
{
private:
   int dim = 2;
   Array<DenseMatrixEigensystem *> EigSystem;
   Array<DenseMatrix *> eigv; // eigenvectors
   Array<DenseMatrix *> SQ;
   mutable Vector dinv;
   void Setup(const Array<DenseMatrix *> & A, const Array<DenseMatrix *> & B);
public:
   FDSolver(const Array<DenseMatrix *> & A, const Array<DenseMatrix *> & B);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~FDSolver();
};

#endif // MFEM_USE_LAPACK

} // mfem name space


#endif // MFEM_FDSOLVER
