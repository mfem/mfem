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

/// @brief Solver for the discontinuous Galerkin mass matrix.
///
/// This class performs a @a local (diagonally preconditioned) conjugate
/// gradient iteration for each element. Optionally, a change of basis is
/// performed to iterate on a better-conditioned system.
///
/// This class fully supports execution on device.
class DGMassInverse : public Solver
{
protected:
   DG_FECollection fec; ///< FE collection in requested basis.
   FiniteElementSpace fes; ///< FE space in requested basis.
   const DofToQuad *d2q; ///< Change of basis. Not owned.
   Array<double> B_; ///< Inverse of change of basis.
   Array<double> Bt_; ///< Inverse of change of basis, transposed.
   class MassIntegrator *m; ///< Owned.
   Vector diag_inv; ///< Jacobi preconditioner.
   double rel_tol = 1e-12; ///< Relative CG tolerance.
   double abs_tol = 1e-12; ///< Absolute CG tolerance.
   int max_iter = 100; ///< Maximum number of CG iterations;

   /// @name Intermediate vectors needed for CG three-term recurrence.
   ///@{
   mutable Vector r_, d_, z_, b2_;
   ///@}

   /// @brief Protected constructor, used internally.
   ///
   /// Custom coefficient and integration rule are used in @a coeff and @a ir
   /// are non-NULL.
   DGMassInverse(FiniteElementSpace &fes_, Coefficient *coeff,
                 const IntegrationRule *ir, int btype);
public:
   /// @brief Construct the DG inverse mass operator for @a fes_.
   ///
   /// The basis type @a btype determines which basis should be used internally
   /// in the solver. This <b>does not</b> have to be the same basis as @a fes_.
   /// The best choice is typically BasisType::GaussLegendre because it is
   /// well-preconditioned by its diagonal.
   ///
   /// The solution and right-hand side used for the solver are not affected by
   /// this basis (they correspond to the basis of @a fes_). @a btype is only
   /// used internally, and only has an effect on the convergence rate.
   DGMassInverse(FiniteElementSpace &fes_, int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff,
                 int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff and IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff,
                 const IntegrationRule &ir, int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(FiniteElementSpace &fes_, const IntegrationRule &ir,
                 int btype=BasisType::GaussLegendre);
   /// Solve the system M b = u. Use @a u as an initial guess.
   void Mult(const Vector &b, Vector &u) const;
   /// Not implemented. Aborts.
   void SetOperator(const Operator &op);
   /// Set the relative tolerance.
   void SetRelTol(const double rel_tol_);
   /// Set the absolute tolerance.
   void SetAbsTol(const double abs_tol_);
   /// Set the maximum number of iterations.
   void SetMaxIter(const double max_iter_);
   ~DGMassInverse();

   // Not part of the public interface, must be public because it contains a
   // kernel
   template<int DIM, int D1D = 0, int Q1D = 0>
   void DGMassCGIteration(const Vector &b_, Vector &u_) const;
};

enum class BatchSolverMode
{
   NATIVE,
   CUSOLVER,
   CUBLAS
};

class DGMassInverse_Direct : public Solver
{
protected:
   const BatchSolverMode mode;
   Vector blocks;
   DenseTensor tensor;
   Array<int> ipiv;

   mutable Array<double*> matrix_array;
   mutable Array<double*> vector_array;
   mutable Array<int> info_array;

   /// @brief Protected constructor, used internally.
   ///
   /// Custom coefficient and integration rule are used in @a coeff and @a ir
   /// are non-NULL.
   DGMassInverse_Direct(FiniteElementSpace &fes_, Coefficient *coeff,
                        const IntegrationRule *ir);
public:
   /// @brief Construct the DG inverse mass operator for @a fes_.
   DGMassInverse_Direct(FiniteElementSpace &fes_,
                        BatchSolverMode mode_ = BatchSolverMode::NATIVE);
   // /// @brief Construct the DG inverse mass operator for @a fes_ with
   // /// Coefficient @a coeff.
   // ///
   // /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   // /// btype.
   // DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff);
   // /// @brief Construct the DG inverse mass operator for @a fes_ with
   // /// Coefficient @a coeff and IntegrationRule @a ir.
   // ///
   // /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   // /// btype.
   // DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff,
   //               const IntegrationRule &ir);
   // /// @brief Construct the DG inverse mass operator for @a fes_ with
   // /// IntegrationRule @a ir.
   // ///
   // /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   // /// btype.
   // DGMassInverse(FiniteElementSpace &fes_, const IntegrationRule &ir,
   //               int btype=BasisType::GaussLegendre);
   /// Solve the system M b = u.
   void Mult(const Vector &b, Vector &u) const;

   void Solve(Vector &u) const;

   /// Not implemented. Aborts.
   void SetOperator(const Operator &op);
};

} // namespace mfem

#endif
