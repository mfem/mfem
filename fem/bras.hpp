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

#ifndef MFEM_BRAS_HPP
#define MFEM_BRAS_HPP

#include "../config/config.hpp"
#include "../linalg/operator.hpp"
#include "../linalg/densemat.hpp"
#include "fespace.hpp"
#include <memory>

namespace mfem
{

/** @brief Broken-space additive Schwarz (BRAS) preconditioner for mass
    matrices and other SPD bilinear forms assembled from element matrices.

    Let $M = P^T M_{br} P$, where $M_{br} = \mathrm{blockdiag}(M_K)$ collects
    the SPD element matrices and $P$ maps true DOFs to element (broken) DOFs.
    The weighted BRAS preconditioner is
    $$ B_W = A_W^{-1} P^T W M_{br}^{-1} W P A_W^{-1}, \qquad A_W = P^T W P, $$
    where $W$ is an element-local SPD weighting. For any such $W$,
    $\lambda_{\min}(B_W M) \geq 1$, and $\lambda_{\max}(B_W M)$ is bounded by
    the Kantorovich factor of the element-wise spectral equivalence between
    $W$ and $M_{br}$. See O. A. Krzysik, B. S. Southworth, and M. Wimmer,
    arXiv:2607.12138.

    Two weightings are supported: $W = I$ (the original BRAS) and the
    recommended $W = \mathrm{diag}(M_{br})$. On conforming meshes $A_W$ is
    diagonal, so applying $B_W$ requires one element-local dense solve per
    element plus the (parallel) assembly $P^T$. The element matrices are
    computed once, at construction or in Update(), and inverted in place.

    The preconditioner acts as the identity on the essential true DOFs. On the
    remaining DOFs it is weighted BRAS for the subspace of functions vanishing
    on the essential DOFs.

    Only domain integrators are supported. Every element must receive an SPD
    contribution from the integrators; otherwise the setup aborts. Variable
    order spaces, NURBS spaces, and meshes with more than one element geometry
    are not supported. */
class BRASPreconditioner : public Solver
{
public:
   /// Choice of the element-local weighting $W$.
   enum class Weighting
   {
      IDENTITY, ///< $W = I$.
      DIAGONAL  ///< $W = \mathrm{diag}(M_{br})$ (recommended).
   };

protected:
   /// The bilinear form defining the element matrices. Not owned, unless
   /// created internally (see @ref owned_form).
   class BilinearForm *form;
   std::unique_ptr<class BilinearForm> owned_form;
   const FiniteElementSpace *fes = nullptr; ///< Not owned.
   Weighting weighting;
   Array<int> ess_tdof_list;

   /// Prolongation from true DOFs to L-DOFs; NULL if it is the identity.
   /// Owned by the space.
   const Operator *P = nullptr;
   /// Element restriction, from L-DOFs to E-DOFs (native ordering). Owned by
   /// the space. The operator $P$ in the class documentation is R P.
   const Operator *R = nullptr;
   /// The element blocks of $W M_{br}^{-1} W$: block e is the nd x nd
   /// matrix WMinvW(:,:,e), where nd is the number of DOFs per element.
   DenseTensor WMinvW;
   /// The inverse of the diagonal of $A_W$, in true DOFs.
   Vector diag_inv;
   /// @ref diag_inv with the essential DOFs set to zero. Scaling by this
   /// vector on both sides gives weighted BRAS for the free DOFs.
   Vector scale;

   /// Work vectors for Mult(): true DOFs (xt), L-DOFs (xl), and the input
   /// and output E-vectors (xe, ye).
   mutable Vector xt, xl, xe, ye;

   /// Build the element blocks of $W M_{br}^{-1} W$ and the diagonal of $A_W$
   /// from the current form and space.
   void Setup();

public:
   /** @brief Construct the BRAS preconditioner for the bilinear form @a a.

       Only the domain integrators of @a a (including their element attribute
       markers) are used, and @a a does not need to be assembled. The form is
       not owned and must remain valid while this object is used. */
   BRASPreconditioner(class BilinearForm &a, const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL);

   /** @brief Construct the BRAS preconditioner for the mass matrix of @a fes.

       The mass integrator is chosen from the space: MassIntegrator for scalar
       spaces, VectorMassIntegrator for scalar spaces with vdim > 1, and
       VectorFEMassIntegrator for vector-valued spaces (e.g. ND and RT). */
   BRASPreconditioner(const FiniteElementSpace &fes,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL);

   /// Same as above, with the scalar coefficient @a Q (not owned).
   BRASPreconditioner(const FiniteElementSpace &fes, Coefficient &Q,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL);

   /** @brief Same as above, with the matrix coefficient @a Q (not owned).

       Valid for vector-valued spaces and for scalar spaces with vdim > 1 (in
       which case the size of @a Q must equal the vdim of @a fes). */
   BRASPreconditioner(const FiniteElementSpace &fes, MatrixCoefficient &Q,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL);

   /// Apply the preconditioner to the true-DOF vector @a x.
   void Mult(const Vector &x, Vector &y) const override;

   /// The preconditioner is symmetric, so this is the same as Mult().
   void MultTranspose(const Vector &x, Vector &y) const override
   { Mult(x, y); }

   /** @brief Does nothing: the preconditioner is defined by the bilinear form
       given at construction, not by @a op.

       This allows using the preconditioner with IterativeSolver, which calls
       SetOperator() on its preconditioner. Call Update() to rebuild the
       preconditioner after the form or its space has changed. */
   void SetOperator(const Operator &op) override { }

   /// Set the list of essential true DOFs, where the preconditioner is the
   /// identity.
   void SetEssentialTrueDofs(const Array<int> &ess_tdof_list);

   /** @brief Rebuild the preconditioner from the bilinear form, keeping the
       current essential true DOFs.

       Call this after the integrator coefficients have changed. If the finite
       element space has changed (e.g. after mesh refinement), use
       Update(const Array<int>&) instead, since the essential true DOFs change
       with the space. */
   void Update();

   /// Rebuild the preconditioner from the bilinear form, with the new list of
   /// essential true DOFs @a ess_tdof_list.
   void Update(const Array<int> &ess_tdof_list);

   /// Return the weighting used by the preconditioner.
   Weighting GetWeighting() const { return weighting; }

   ~BRASPreconditioner();
};

} // namespace mfem

#endif
