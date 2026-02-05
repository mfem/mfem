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

#ifndef MFEM_DGMASSINV_HPP
#define MFEM_DGMASSINV_HPP

#include "../linalg/operator.hpp"
#include "fespace.hpp"
#include "kernel_dispatch.hpp"
#include <memory>

namespace mfem
{

/// @brief Solver for the discontinuous Galerkin mass matrix.
///
/// This class performs a @a local (diagonally preconditioned) conjugate
/// gradient iteration for each element. Optionally, a change of basis is
/// performed to iterate on a better-conditioned system. This class fully
/// supports execution on device (GPU).
class DGMassInverse : public Solver
{
protected:
   DG_FECollection fec; ///< FE collection in requested basis.
   FiniteElementSpace fes; ///< FE space in requested basis.
   const DofToQuad *d2q; ///< Change of basis. Not owned.
   Array<real_t> B_; ///< Inverse of change of basis.
   Array<real_t> Bt_; ///< Inverse of change of basis, transposed.
   std::unique_ptr<class BilinearForm> M; ///< Mass bilinear form.
   class MassIntegrator *m; ///< Mass integrator, owned by the form @ref M.
   Vector diag_inv; ///< Jacobi preconditioner.
   real_t rel_tol = 1e-12; ///< Relative CG tolerance.
   real_t abs_tol = 1e-12; ///< Absolute CG tolerance.
   int max_iter = 100; ///< Maximum number of CG iterations;

   /// @name Intermediate vectors needed for CG three-term recurrence.
   ///@{
   mutable Vector r_, d_, z_, b2_;
   ///@}

   /// @brief Protected constructor, used internally.
   ///
   /// Custom coefficient and integration rule are used if @a coeff and @a ir
   /// are non-NULL.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient *coeff,
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
   DGMassInverse(const FiniteElementSpace &fes_,
                 int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                 int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff and IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                 const IntegrationRule &ir, int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, const IntegrationRule &ir,
                 int btype=BasisType::GaussLegendre);
   /// @brief Solve the system M b = u.
   ///
   /// If @ref iterative_mode is @a true, @a u is used as an initial guess.
   void Mult(const Vector &b, Vector &u) const override;
   /// Same as Mult() since the mass matrix is symmetric.
   void MultTranspose(const Vector &b, Vector &u) const override { Mult(b, u); }
   /// Not implemented. Aborts.
   void SetOperator(const Operator &op) override;
   /// Set the relative tolerance.
   void SetRelTol(const real_t rel_tol_);
   /// Set the absolute tolerance.
   void SetAbsTol(const real_t abs_tol_);
   /// Set the maximum number of iterations.
   void SetMaxIter(const int max_iter_);
   /// Recompute operator and preconditioner (when coefficient or mesh changes).
   void Update();

   ~DGMassInverse();

   /// @brief Solve the system M b = u. <b>Not part of the public interface.</b>
   /// @note This member function must be public because it defines an
   /// extended lambda used in an mfem::forall kernel (nvcc limitation)
   template<int DIM, int D1D = 0, int Q1D = 0>
   void DGMassCGIteration(const Vector &b_, Vector &u_) const;

   using CGKernelType = void(DGMassInverse::*)(const Vector &b_, Vector &u) const;
   MFEM_REGISTER_KERNELS(CGKernels, CGKernelType, (int, int, int));
};

} // namespace mfem

#endif
