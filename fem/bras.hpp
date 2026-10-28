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
#ifdef MFEM_USE_MPI
#include "../linalg/hypre.hpp"
#endif
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

    On nonconforming meshes, $P = P_0 E$, where $P_0$ is a signed Boolean
    matrix and $E$ interpolates the hanging (slave) DOFs from their master
    DOFs. Then $A_W$ is no longer diagonal and its inverse is not sparse. The
    rows of $E$ split into the rows with at most one entry (J) and the rows
    with more than one entry (H, forming the matrix C), and
    $A_W = \Lambda_u + C^T \Lambda_h C$ with diagonal $\Lambda_u$ and
    $\Lambda_h$. The action of $A_W^{-1}$ is then approximated with one of the
    methods in NCMethod; see SetNCMethod(). The same holds for parallel
    Nedelec spaces with DOFs on shared triangular faces, whose prolongation
    has rows with two entries (a change of basis between the face DOFs seen
    by the two ranks).

    The preconditioner acts as the identity on the essential true DOFs. On the
    remaining DOFs it is weighted BRAS for the subspace of functions vanishing
    on the essential DOFs.

    The element blocks of $W M_{br}^{-1} W$ are either stored as dense
    matrices or, for mass matrices on affine tensor product elements, applied
    by sum factorization; see ElementInverse.

    Mult() runs on the device, if one is configured (see Device). The setup
    (construction, Update(), SetEssentialTrueDofs() and SetNCMethod()) and
    the assembly methods run on the host.

    Only domain integrators are supported. Every element must receive an SPD
    contribution from the integrators; otherwise the setup aborts. Variable
    order spaces, NURBS spaces, and meshes with more than one element geometry
    are not supported.

    See the miniapps bras_mass and bras_darcy in miniapps/solvers. */
class BRASPreconditioner : public Solver
{
public:
   /// Choice of the element-local weighting $W$.
   enum class Weighting
   {
      IDENTITY, ///< $W = I$.
      DIAGONAL  ///< $W = \mathrm{diag}(M_{br})$ (recommended).
   };

   /// Computation and application of the element blocks of
   /// $W M_{br}^{-1} W$.
   enum class ElementInverse
   {
      /** The element matrices are computed by the integrators, inverted with
          a Cholesky factorization, and stored as dense blocks of size
          $n_d \times n_d$, where $n_d$ is the number of DOFs per element.
          Supports all the integrators, spaces and meshes of the class
          (default). */
      DENSE,
      /** Sum factorization. On an affine tensor product element K with a
          constant coefficient, the element mass matrix of each vector
          component is a multiple $m_K$ of the Kronecker product of 1D
          reference mass matrices, so $M_K^{-1}$ and $\mathrm{diag}(M_K)$
          are Kronecker products as well. Only one number per element and
          component is stored, and each element block is applied with
          $O(p^{d+1})$ operations instead of $O(p^{2d})$.

          The requirements are checked in the setup, which aborts if they do
          not hold:
          - The mesh consists of segments, quadrilaterals or hexahedra, all
            with affine transformations, and has the dimension of the space.
          - The space is an H1 or L2 space (any vdim), or an ND or RT space,
            with tensor product elements (e.g. not serendipity elements).
            For ND and RT spaces, the columns of the Jacobian must also be
            orthogonal on each element (rectangles and boxes, possibly
            rotated); otherwise the element matrix couples the vector
            components and is not a Kronecker product.
          - The domain integrators are mass integrators matching the space:
            MassIntegrator (vdim = 1), VectorMassIntegrator (vdim > 1) or
            VectorFEMassIntegrator (ND and RT), with no coefficient or with a
            scalar Coefficient. Attribute markers and custom integration rules
            are not supported.

          The coefficients are evaluated at the center of each element. For
          piecewise constant coefficients (e.g. PWConstCoefficient) the
          preconditioner then equals the one of ElementInverse::DENSE, up to
          rounding. For other coefficients it is the BRAS preconditioner of
          the mass matrix with the coefficient replaced by its value at the
          element centers. */
      TENSOR
   };

   /** @brief Approximation of $A_W^{-1}$ on nonconforming spaces. All methods
       reduce to $B_W$ on conforming spaces.

       With $S = \mathrm{diag}(A_W)^{-1}$ and
       $G = P^T W M_{br}^{-1} W P$, the methods are: */
   enum class NCMethod
   {
      /** The left inverse $B_X = R_X M_{br}^{-1} R_X^T$, where $R_X$ is the
          left inverse of $P$ (i.e. $R_X P = I$) obtained from the exact one,
          $A_W^{-1} P^T W$, by replacing $A_W^{-1}$ by $S$ in the coupling
          $X = A_W^{-1} C^T \Lambda_h$ of the hanging DOFs. Since $R_X$ is a
          left inverse, $\lambda_{\min}(B_X M) \geq 1$. Each application
          requires three products with $P_L$ and three with $P_L^T$
          (recommended). */
      LEFT_INVERSE,
      /** $q_k(A_W) G q_k(A_W)$, where $q_k(A_W) \approx A_W^{-1}$ is the
          Chebyshev polynomial approximation of degree k (in $S A_W$) for the
          spectral interval [a, b] of $S A_W$. In the setup, a is set to a
          lower bound, $\min (\Lambda_u S)$, and b to an estimate from a few
          power iterations. As k increases the preconditioner converges to
          $B_W$; k = 2 is usually close to it. Each application requires 2k
          products with $A_W$, i.e. 2k + 1 products with $P_L$ and with
          $P_L^T$. */
      CHEBYSHEV
   };

protected:
   /// The bilinear form defining the element matrices. Not owned, unless
   /// created internally (see @ref owned_form).
   class BilinearForm *form;
   std::unique_ptr<class BilinearForm> owned_form;
   const FiniteElementSpace *fes = nullptr; ///< Not owned.
   Weighting weighting;
   ElementInverse element_inverse;
   Array<int> ess_tdof_list;

   /// Prolongation from true DOFs to L-DOFs; NULL if it is the identity.
   /// Owned by the space.
   const Operator *P = nullptr;
   /// Element restriction, from L-DOFs to E-DOFs (native ordering, or
   /// lexicographic ordering for ElementInverse::TENSOR). Owned by the space.
   /// The operator $P$ in the class documentation is R P.
   const Operator *R = nullptr;
   /// Only for ElementInverse::DENSE: the element blocks of
   /// $W M_{br}^{-1} W$: block e is the nd x nd matrix WMinvW(:,:,e), where
   /// nd is the number of DOFs per element.
   DenseTensor WMinvW;
   /// Only for ElementInverse::TENSOR: block e of $W M_{br}^{-1} W$ is, for
   /// each vector component c, tensor_scale(c + ncomp*e) times the Kronecker
   /// product of the 1D factors of c. The 1D factor of component c in
   /// direction d is the column-major tensor_n[c][d] x tensor_n[c][d] matrix
   /// starting at entry tensor_offset[c][d] of tensor_1d. The sizes are 1 in
   /// the directions beyond the mesh dimension.
   Vector tensor_1d, tensor_scale;
   int tensor_ncomp = 0;
   int tensor_n[3][3], tensor_offset[3][3];
   /// The inverse of the diagonal of $A_W$, in true DOFs.
   Vector diag_inv;
   /// @ref diag_inv with the essential DOFs set to zero. Scaling by this
   /// vector on both sides gives weighted BRAS for the free DOFs.
   Vector scale;

   NCMethod nc_method = NCMethod::LEFT_INVERSE;
   int cheb_degree = 2;
   /// True if $P_L$ has rows with more than one entry (the H rows) on any
   /// rank. Then $A_W$ is not diagonal and @ref nc_method is used.
   bool nc = false;
   /// The diagonal of $D_W = P_0^T W P_0$ (an L-vector): the sum of the
   /// weights of the element DOFs of each L-DOF.
   Vector dl;
   /// Only if @ref nc: the L-vector equal to 1 on the J rows of $P_L$ and 0
   /// on the H rows, and the L-vector equal to @ref dl on the H rows (that
   /// is, $\Lambda_h$) and 0 on the J rows.
   Vector jmask, hdl;
   /// Only if @ref nc: the inverse of $\Lambda_u$ (a T-vector), and the same
   /// with the essential DOFs set to zero.
   Vector lu_inv, lu_scale;
   /// Only if @ref nc: an interval [a, b] containing the spectrum of
   /// $S A_W$, used by NCMethod::CHEBYSHEV.
   real_t cheb_a = 1.0, cheb_b = 1.0;

   /// Work vectors for Mult(): true DOFs (xt, ut), L-DOFs (xl, zl), the
   /// input and output E-vectors (xe, ye), and the Chebyshev iteration
   /// (cr, cd, cq).
   mutable Vector xt, ut, xl, zl, xe, ye, cr, cd, cq;

   /// Build the element blocks of $W M_{br}^{-1} W$ and the prolongation data
   /// from the current form and space.
   void Setup();

   /// Compute the element blocks of $W M_{br}^{-1} W$ for ElementInverse::
   /// DENSE and ElementInverse::TENSOR, and set the E-vector w to the
   /// diagonal of W.
   void SetupDense(Vector &w);
   void SetupTensor(Vector &w);

   /// Compute @ref diag_inv, and the data of the nonconforming methods, from
   /// @ref dl and the prolongation.
   void SetupProlongation();

   /// Compute the interval @ref cheb_a, @ref cheb_b for the free DOFs.
   void SetupChebyshev();

   /// y = R_E^T (W M_br^{-1} W) R_E x, for L-vectors x and y (which may be
   /// the same vector).
   void ApplyBroken(const Vector &x, Vector &y) const;

   /// Apply the element blocks of ElementInverse::TENSOR to the E-vector xe.
   /// Uses xe and ye, and returns the one holding the result.
   const Vector &ApplyTensor() const;

   /// y = A_W x = P_L^T D_W P_L x, for T-vectors.
   void ApplyAW(const Vector &x, Vector &y) const;

public:
   /// @name Internal methods
   /// Public only because they contain mfem::forall kernels: CUDA does not
   /// allow extended lambdas in private or protected member functions.
   ///@{

   /// x = q_k(A_W) b on the free DOFs, the Chebyshev approximation of
   /// A_W^{-1} b; x is zero on the essential DOFs.
   void ApplyChebyshev(const Vector &b, Vector &x) const;

   /// Apply the NCMethod::LEFT_INVERSE method on the free DOFs; y is zero on
   /// the essential DOFs.
   void MultLeftInverse(const Vector &x, Vector &y) const;
   ///@}

protected:
   /// Apply the NCMethod::CHEBYSHEV method on the free DOFs; y is zero on the
   /// essential DOFs.
   void MultChebyshev(const Vector &x, Vector &y) const;

   /// Return the local matrix R_E^T (W M_br^{-1} W) R_E on the L-DOFs, owned
   /// by the caller.
   SparseMatrix *AssembleBroken() const;

public:
   /** @brief Construct the BRAS preconditioner for the bilinear form @a a.

       Only the domain integrators of @a a (including their element attribute
       markers) are used, and @a a does not need to be assembled. The form is
       not owned and must remain valid while this object is used. */
   BRASPreconditioner(class BilinearForm &a, const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL,
                      ElementInverse element_inverse =
                         ElementInverse::DENSE);

   /** @brief Construct the BRAS preconditioner for the mass matrix of @a fes.

       The mass integrator is chosen from the space: MassIntegrator for scalar
       spaces, VectorMassIntegrator for scalar spaces with vdim > 1, and
       VectorFEMassIntegrator for vector-valued spaces (e.g. ND and RT). */
   BRASPreconditioner(const FiniteElementSpace &fes,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL,
                      ElementInverse element_inverse =
                         ElementInverse::DENSE);

   /// Same as above, with the scalar coefficient @a Q (not owned).
   BRASPreconditioner(const FiniteElementSpace &fes, Coefficient &Q,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL,
                      ElementInverse element_inverse =
                         ElementInverse::DENSE);

   /** @brief Same as above, with the matrix coefficient @a Q (not owned).

       Valid for vector-valued spaces and for scalar spaces with vdim > 1 (in
       which case the size of @a Q must equal the vdim of @a fes). */
   BRASPreconditioner(const FiniteElementSpace &fes, MatrixCoefficient &Q,
                      const Array<int> &ess_tdof_list,
                      Weighting weighting = Weighting::DIAGONAL,
                      ElementInverse element_inverse =
                         ElementInverse::DENSE);

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
   /// identity. In parallel, this must be called on all ranks.
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

   /// Return the computation of the element blocks used by the
   /// preconditioner.
   ElementInverse GetElementInverse() const { return element_inverse; }

   /** @brief Set the approximation of $A_W^{-1}$ used on nonconforming
       spaces, and the polynomial degree (at least 1) used by
       NCMethod::CHEBYSHEV. The default is NCMethod::LEFT_INVERSE. Has no
       effect on conforming spaces, where $A_W$ is diagonal. In parallel, this
       must be called on all ranks. */
   void SetNCMethod(NCMethod method, int degree = 2);

   /// Return the approximation of $A_W^{-1}$ used on nonconforming spaces.
   NCMethod GetNCMethod() const { return nc_method; }

   /** @brief Return the preconditioner as a sparse matrix, owned by the
       caller. Only for serial spaces; see ParallelAssembleMatrix().

       The matrix equals the operator applied by Mult(), including the
       identity on the essential true DOFs. On conforming spaces it is
       $A_W^{-1} P^T W M_{br}^{-1} W P A_W^{-1}$ with diagonal $A_W$, so it has
       the sparsity pattern of $M$. On nonconforming spaces, NCMethod::
       LEFT_INVERSE gives $B_X = Q^T R_E^T W M_{br}^{-1} W R_E Q$, where the
       sparse matrix $Q = R_X^T$ has the shape of $P_L$ with a wider stencil
       at the hanging DOFs. NCMethod::CHEBYSHEV cannot be assembled, since its
       matrix is not sparse; then this method aborts. */
   SparseMatrix *AssembleMatrix() const;

#ifdef MFEM_USE_MPI
   /** @brief Return the preconditioner as a parallel matrix, owned by the
       caller. Only for parallel spaces; the matrix is the one described in
       AssembleMatrix(). Must be called on all ranks. */
   HypreParMatrix *ParallelAssembleMatrix() const;
#endif

   ~BRASPreconditioner();
};

} // namespace mfem

#endif
