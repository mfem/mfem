// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DG_MASS_INVERSE
#define MFEM_DG_MASS_INVERSE

#include "../linalg/operator.hpp"
#include "fespace.hpp"

namespace mfem
{

// class Coefficient;
class MassIntegrator;

class DGMassInverse : public Solver
{
protected:
   DG_FECollection fec; ///< FE collection in requested basis.
   FiniteElementSpace fes; ///< FE space in requested basis.
   const DofToQuad *d2q; ///< Change of basis. Not owned.
   MassIntegrator *m; ///< Owned.
   Vector diag_inv; ///< Jacobi preconditioner.
   double rel_tol = 1e-12; ///< Relative CG tolerance.
   double abs_tol = 1e-12; ///< Absolute CG tolerance.
   int max_iter = 100; ///> Maximum number of CG iterations;

   /// @name Intermediate vectors needed for CG three-term recurrence.
   ///@{
   mutable Vector r_, d_, z_;
   ///@}

public:
   DGMassInverse(FiniteElementSpace &fes_, Coefficient *coeff,
                 int btype=BasisType::GaussLegendre);
   DGMassInverse(FiniteElementSpace &fes_, int btype=BasisType::GaussLegendre);
   void Mult(const Vector &Mu, Vector &u) const;
   void SetOperator(const Operator &op);
   void SetRelTol(const double rel_tol_);
   void SetAbsTol(const double abs_tol_);
   void SetMaxIter(const double max_iter_);
   ~DGMassInverse();

   // Not part of the public interface, must be public because it contains a
   // kernel
   template<int DIM, int D1D = 0, int Q1D = 0>
   void DGMassCGIteration(const Vector &b_, Vector &u_) const;
};

} // namespace mfem

#endif
