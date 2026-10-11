// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SCHWARZ
#define MFEM_SCHWARZ

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "hypre.hpp"
#include "densemat.hpp"
#include <memory>
#include <vector>

namespace mfem
{

/**
* @class AdditiveSchwarz
* @brief Additive Schwarz smoother for a HypreParMatrix with user-defined,
* possibly overlapping, subdomains that are solved with dense LU
* factorizations.
*
* Given subdomains $ S_1, \dots, S_m $, each a set of global true dofs (global
* row indices of the operator $ A $), the smoother is
*  $$
* M = \sum_s w_s R_s^T A_s^{-1} R_s, \qquad A_s = R_s A R_s^T,
*  $$
* where $ R_s $ is the boolean restriction onto the dofs of $ S_s $. The weight
* $ w_s $ is the reciprocal of the average number of subdomains per dof of
* $ S_s $,
*  $$
* w_s = \frac{|S_s|}{\sum_{i \in S_s} c_i},
*  $$
* where $ c_i $ is the number of subdomains, across all MPI ranks, that
* contain dof $ i $. Dofs that are not in any subdomain receive no correction.
*
* The work is distributed across MPI ranks by subdomain: each rank assembles,
* factors and applies exactly the subdomains passed to SetSubdomains() on that
* rank. A subdomain may contain dofs owned by any rank; the required rows of
* @a A, the restriction of the input, and the summation of the corrections are
* all communicated through hypre.
*
* SetSubdomains() and SetOperator() are collective, and the smoother is
* (re)built whenever either is called once both have been set.
*
* The operator @a A is not owned: it must remain valid, and must not be
* modified, while the smoother is in use (call SetOperator() again to rebuild
* the smoother after changing it). The subdomain lists are copied.
*
* See the `-amgf-schwarz` option of the contact miniapp
* (miniapps/contact/contact.cpp) for an example of use as the subspace solver
* of AMGFSolver.
*/
class AdditiveSchwarz : public Solver
{
public:
   /// Construct an empty smoother. SetSubdomains() and SetOperator() must be
   /// called before use.
   AdditiveSchwarz() : Solver() { }

   /// Construct the smoother for @a A_ with the given @a subdomains_; see
   /// SetSubdomains(). The operator @a A_ is not owned and must outlive the
   /// smoother.
   AdditiveSchwarz(const HypreParMatrix &A_,
                   const std::vector<Array<HYPRE_BigInt>> &subdomains_);

   /**
   * @brief Set the subdomains owned by this MPI rank.
   *
   * Each entry of @a subdomains_ lists the global true dofs (global row
   * indices of the operator) of one subdomain. The dofs may be owned by any
   * rank, may appear in any order, and may be shared with any number of other
   * subdomains on this or other ranks. Repeated dofs within one subdomain are
   * counted once. Ranks that own no subdomain must still call this method,
   * with an empty list. The lists are copied, so @a subdomains_ need not
   * outlive this call.
   */
   void SetSubdomains(const std::vector<Array<HYPRE_BigInt>> &subdomains_);

   /// Set the operator, which must be a square HypreParMatrix. The operator
   /// is not owned and must outlive the smoother.
   void SetOperator(const Operator &op) override;

   /// Apply the smoother: $ y = M x $, or $ y \mathrel{+}= M (x - A y) $ if
   /// iterative_mode is true.
   void Mult(const Vector &x, Vector &y) const override;

   /// Apply the transposed smoother: $ y = M^T x $, or
   /// $ y \mathrel{+}= M^T (x - A^T y) $ if iterative_mode is true.
   void MultTranspose(const Vector &x, Vector &y) const override;

private:
   /// System operator (not owned).
   const HypreParMatrix *A = nullptr;
   /// Global dofs of the local subdomains, each sorted and without repeats.
   std::vector<Array<HYPRE_BigInt>> subdomains;
   /// True once SetSubdomains() has been called (possibly with an empty list).
   bool subdomains_set = false;
   /// Boolean restriction from the global dofs onto the concatenated local
   /// subdomain dofs; subdomain s occupies rows [offsets[s], offsets[s+1])
   /// (owned).
   std::unique_ptr<HypreParMatrix> R;
   /// Offsets of the local subdomains in the rows of R, of size
   /// subdomains.size() + 1.
   Array<int> offsets;
   /// Correction weight of each local subdomain.
   Vector weights;
   /// LU factors of each local subdomain matrix, with pivots of subdomain s
   /// stored in ipiv starting at offsets[s].
   std::vector<DenseMatrix> lu;
   Array<int> ipiv;

   /// Work vector for the restricted residual and the local corrections, of
   /// size offsets.Last().
   mutable Vector xs;
   /// Work vector for the residual when iterative_mode is true.
   mutable Vector r;

   /// Build R and the weights, then assemble and factor the local subdomain
   /// matrices.
   void Setup();
   /// Mult() and MultTranspose() implementation.
   void Apply(const Vector &x, Vector &y, bool transpose) const;
};

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
