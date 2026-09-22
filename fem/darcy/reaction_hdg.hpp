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

#ifndef MFEM_DARCY_REACTION_HDG
#define MFEM_DARCY_REACTION_HDG

#include "../../config/config.hpp"
#include "../nonlininteg.hpp"
#include "postprocess_hdg.hpp"

namespace mfem
{

/** @brief A pointwise reaction law, evaluated at nodal values.

    Nodal, not quadrature: this is the interface that makes the method
    interpolatory. It is asked for a value at ONE node and knows nothing about
    integration, so the same object serves both the interpolatory term and the
    quadrature control that exists to measure it. */
class NodalReactionFunction
{
public:
   virtual ~NodalReactionFunction() = default;

   virtual int NumEquations() const { return 1; }

   /// `F(u)` at one point. @a x is that point's physical coordinate.
   virtual void Eval(const Vector &x, const Vector &u, Vector &F) const = 0;

   /** @brief `dF/du` at the same point, `neq x neq`.

       **Not optional.** In a hybridized method the Jacobian is never
       assembled globally, so a missing piece of it gives no wrong answer --
       only a Newton that misbehaves, which is a failure that survives a
       passing regression suite. This branch has paid for exactly that twice.*/
   virtual void EvalJacobian(const Vector &x, const Vector &u,
                             DenseMatrix &J) const = 0;
};

/** @brief Common machinery for a reaction term evaluated at the postprocessed
    potential: the postprocessing blocks, `A9`, and the element node images.

    Both concrete integrators below form `gamma = B11 u_e + B12 p_e` through
    the SAME HDGPostprocessBlocks and differ only in what they then do with
    `F`. That is the point of the split: they are two quadratures of one
    discrete object, so a disagreement between them is a statement about the
    interpolation and not about the postprocessing.

    The blocks are BORROWED and must outlive this object; one instance may be
    shared by any number of reaction integrators, so the geometry pass is paid
    once. */
class HDGReactionIntegratorBase : public BlockNonlinearFormIntegrator
{
protected:
   const NodalReactionFunction *F;
   HDGPostprocessBlocks *blocks;
   int neq{0};
   int ir_order{-1};

   /// `A9(i,j) = (chi_j, phi_i)_K`, `nd x ns` per element.
   Array<int> offs_A9, offs_x;
   Array<real_t> A9_data;
   /// The physical images of the enriched element's nodes, `ns*dim` per
   /// element. `Z_h`'s nodes are fixed on the reference element, so their
   /// images are geometry and belong with the matrices rather than in the
   /// hot loop.
   Array<real_t> x_data;
   bool assembled{false};

   void CheckFresh() const;

   /** @brief `gamma` for one element, from the blocks. The ONE place it is
       formed, so the residual and the gradient cannot be of different
       operators.

       Deliberately recomputed in both entry points rather than cached between
       them: a cached `gamma` would need invalidating every Newton step, which
       is the class of run-time reconstruction this formulation exists to
       delete. It is one GEMV against the quadrature of `F` it replaces. */
   void Prepare(int el, const Vector &u_l, const Vector &p_l,
                Vector &gamma) const;

public:
   HDGReactionIntegratorBase(const NodalReactionFunction &F_,
                             HDGPostprocessBlocks &blocks_);

   /// Quadrature order for `A9`; negative asks for one suited to `Z_h`.
   void SetIntegrationOrder(int order) { ir_order = order; }

   /** @brief Form `A9` and the node images for every element. Idempotent, and
       the only place this class reads geometry or a quadrature rule. Calls
       the blocks' own Assemble() if it has not run. */
   void Assemble();

   /** @brief The potential row and nothing else.

       `(I_h F(u*_h), v)_K` is tested against the POTENTIAL space only, so
       the flux row is untouched and the flux mass stays linear. The row
       READS the flux -- `gamma = B11 u_e + B12 p_e` -- and that is a
       different statement: the promise is about what is written, and the
       reader of the promise carries the (1,0) block through
       DarcyHybridization's Bg_data. */
   int GetBlockRowMask() const override { return 1 << 1; }

   /// `A9` for element @a el, `nd x ns`.
   void GetA9(int el, DenseMatrix &A9) const;
   /// Physical coordinates of element @a el's enriched nodes, `ns x dim`.
   void GetNodes(int el, DenseMatrix &x) const;
};

/** @brief CCSZ-I's interpolatory reaction term, `(I_h F(u*_h), v)_K`.

    The residual is `A9 F(gamma)` and the gradient is
    `A9 diag(F'(gamma)) [B11  B12]`, with `gamma = B11 u_e + B12 p_e`. All of
    `A9`, `B11` and `B12` are assembled once; `F` and `F'` are nodal
    evaluations, so **no quadrature of `F` happens in any residual or Jacobian
    evaluation**. That is the saving the method exists for.

    Chen, Cockburn, Singler & Zhang, J. Sci. Comput. 81 (2019) 2188-2212,
    §2.2 and the equations after (9).

    @note This is a DISCRETISATION change and not a quadrature refinement.
    `I_h F(u*) = F(u*)` only when `F(u*)` lies in `Z_h`, which for a constant
    `a + c u` it does and for anything else it does not -- see
    HDGQuadratureReactionIntegrator, which is the control that measures the
    difference rather than a slower route to the same answer.

    @warning **On a tensor-product element with MFEM's default L2 basis this
    term IS the quadrature term, identically, for every `F`.** L2_FECollection
    is nodal at the GAUSS-LEGENDRE points, so the enriched space's `k+2` points
    per dimension are exactly the points of the rule the postprocessing
    already uses, and `A9` built with that rule gives
    `(chi_j, phi_i) = w_j phi_i(x_j)`. Interpolation has degenerated to
    collocation and there is nothing left to measure. Measured: relative gap
    4.3e-16 on a box against 1.2e-03 on a triangle at the same order and rule,
    and 6.5e-03 on the same box with a Gauss-Lobatto enriched basis. Anyone
    comparing the two forms must break the coincidence deliberately -- a
    simplex, a non-Gauss basis, or a finer rule on the control -- or the
    comparison passes for any implementation whatsoever.

    @note **In a converged solve the difference is not visible.** On the steady
    problem `Delta u - (u^3 - u) = g` on triangles at O(1) `tau`, this term and
    the quadrature control give the same rates and agree in the postprocessed
    error to the FIFTH digit at k = 1 (7.26499e-06 against 7.26462e-06 at
    n = 32) and the sixth at k = 2. So the saving is free: Remark 2.3's claim
    holds here. Rates measured for `u*`: 1.00, 3.00, 4.01, 5.00 at k = 0..3,
    i.e. `k+2` for k >= 1 and no superconvergence at k = 0, which is CCSZ's
    Table 1 (3.01 and 0.97).

    @note And where the two DO differ, the gap converges two orders faster
    than the interpolation error suggests: `h^{k+4}` on a Gauss-Legendre box
    (measured 5.00, 6.00, 7.00 at k = 1, 2, 3) rather than the `h^{k+2}` of
    `||I_h g - g||`. The leading interpolation error is the degree-`k+2`
    Legendre polynomial, which is L2-orthogonal to `P^{k+1}`, and since
    `(x-x_c) phi` lies in `P^{k+1}` for `phi` in `W_h` the next term vanishes
    too. A Gauss-Lobatto basis recovers `h^{k+2}` exactly (measured 2.99,
    4.00), which is what identified the mechanism.

    @warning **A discontinuity in `F` inside an element is not seen AT ALL
    unless it separates two of the nodes.** Gate the reaction off across a
    plane (`convdiff -p 10 -rcm 1 -rc a`) and hold the plane inside the
    outermost enriched node -- 0.065 of a cell width at k = 2 -- and every
    printed digit of every norm is identical to a run with the plane ON a mesh
    line, checked by diffing two whole runs. The method then reports full
    `k+2` superconvergence for a problem whose interface it has silently
    snapped to the cell edge, so an accurate-looking rate is NOT evidence that
    an interface was resolved. Nothing here can detect it; the caller has to
    know where the interface is.

    @note **Where the discontinuity IS seen it costs `O(h)` whatever the
    degree, and HDGQuadratureReactionIntegrator is not a repair.** On
    triangles at `tau = 1`: a jump lying ON a mesh line costs nothing at all
    (the smooth ladder to six digits at k = 1, 2 and 3), while a jump cutting
    a column of cells, held at a fixed fraction of a cell as `h` halves, gives
    `u*` rate 1.00 -- 0.88 to 1.20 at k = 1, 1.00 to 1.01 at k = 2 -- and 990x
    the uncut error at `h = 1/32`, k = 2. Sweeping the interface across one
    cell in 32 steps, the error is a STAIRCASE whose plateau boundaries are
    THIS class's nodes for the interpolatory arm and the control's quadrature
    points for the other. Of 29 cut positions the control is smaller at 15 and
    larger at 14, and the ratio spans 0.013 to 369: which arm wins is decided
    by an accident of geometry rather than by the discretisation.
    **That is why this class carries no per-element opt-out.** Swapping to
    quadrature on the cut elements -- the repair a caller would ask for --
    trades one arbitrary point set for another and opts into the dearer arm.
    What a cut element needs is a rule that knows where the cut is, which is
    an unfitted-quadrature machine and not a flag on an integrator. The
    numbers are in `miniapps/hdg/convdiff.cpp`'s header comment.

    @note A gate on `F` has a delta for its derivative, and a caller who drops
    it hands Newton a one-sided Jacobian. Measured: 3 to 6 outer steps against
    4 to 5 ungated, i.e. nothing structural. What a MOVING interface costs is
    LOCAL work, the local nonlinear solve chasing a discontinuity that shifts
    with the iterate. */
class HDGInterpolatoryReactionIntegrator : public HDGReactionIntegratorBase
{
public:
   using HDGReactionIntegratorBase::HDGReactionIntegratorBase;

   /// `A9 F(gamma)` into `elvec[1]`; `elvec[0]` is emptied -- no flux row.
   void AssembleElementVector(const Array<const FiniteElement*> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector*> &elfun,
                              const Array<Vector*> &elvec) override;

   /// `A9 diag(F'(gamma)) B11` into `elmats(1,0)` and `... B12` into
   /// `elmats(1,1)`. `(0,0)` and `(0,1)` are emptied.
   void AssembleElementGrad(const Array<const FiniteElement*> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector*> &elfun,
                            const Array2D<DenseMatrix*> &elmats) override;
};

/** @brief `(F(u*_h), v)_K` by quadrature -- the control for the interpolatory
    term, and the only reference the tree has.

    There is no reaction integrator anywhere else in this library, so without
    this one a claim about what interpolation costs would have nothing to be
    measured against. It evaluates `u*` at quadrature points from the same
    `gamma` and integrates `F(u*)` against the potential basis, so the two
    differ in the treatment of `F` ALONE.

    Slower by construction and not meant for production: it pays quadrature of
    `F` and `F'` on every call, which is exactly what the interpolatory form
    removes. */
class HDGQuadratureReactionIntegrator : public HDGReactionIntegratorBase
{
public:
   using HDGReactionIntegratorBase::HDGReactionIntegratorBase;

   void AssembleElementVector(const Array<const FiniteElement*> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector*> &elfun,
                              const Array<Vector*> &elvec) override;

   void AssembleElementGrad(const Array<const FiniteElement*> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector*> &elfun,
                            const Array2D<DenseMatrix*> &elmats) override;
};

}

#endif
