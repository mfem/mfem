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

#ifndef MFEM_DARCY_POSTPROCESS_HDG
#define MFEM_DARCY_POSTPROCESS_HDG

#include "../../config/config.hpp"
#include "../gridfunc.hpp"
#include "../coefficient.hpp"
#include "../../linalg/densemat.hpp"

namespace mfem
{

/** @brief The FIXED element-local blocks of the classic HDG postprocessing.

    On each element the nodal values of the postprocessed potential are

        gamma_e = B11_e u_e + B12_e p_e

    where `u_e` and `p_e` are that element's flux and potential coefficients
    for one equation, `B11_e` is `ns x na`, and `B12_e` is `ns x nd` and
    **rank one** -- `u*` sees the potential only through its element average.
    `ns`, `na` and `nd` are the enriched, flux and potential dof counts of the
    element, per equation.

    **Constant in the state by construction.** Only the geometry and the
    inverse diffusivity enter, so Assemble() runs once and nothing invalidates
    it when the solution moves, or when a parameter of a reaction term moves.
    That is what makes an interpolatory formulation cheap: the quadrature is
    paid once rather than once per residual evaluation.

    What DOES invalidate it is a moving mesh or a moving diffusion
    coefficient, because the inverse diffusivity is a right-hand-side operator
    of the local problem. Neither is guarded silently: Assemble() records the
    Mesh::sequence it saw, the caller announces a coefficient move with
    CoefficientsMoved(), and Apply() and GetBlocks() abort on a mismatch
    rather than return a stale answer.

    **Systems.** Everything is per equation and the blocks do not depend on
    which equation they are applied to, so one pair is stored per element and
    used `neq` times. Vectors are laid out equation outermost throughout,
    as the rest of the branch lays blocks out.

    @note **No integrator list reaches this class, and that is correct rather
    than an omission.** CCSZ-I's `u*` solves a local problem in which a
    reaction term does not appear at all -- the reaction is evaluated AT its
    nodes, downstream of it -- so there is nothing here for a reaction
    integrator to be added to or dropped from. The whitelist that does exist,
    in DarcyForm::ReconstructFluxAndPot(), belongs to the richer mixed
    reconstruction and admits only the two convection integrators; it never
    sees this path, and a CCSZ-I term is a BlockNonlinearFormIntegrator on the
    block nonlinear form, which that loop does not consult either.

    @note The closure used here is the row replacement of
    HDGPotentialPostprocessor -- one row of the Neumann stiffness is
    overwritten by the element mean -- and NOT the bordered saddle-point form
    in which the method is usually written. The two agree in exact arithmetic,
    because the rows of the local stiffness sum to zero and the flux data is
    consistent, but the blocks that come out are this system's. A sibling
    implementing the bordered form would not get these matrices entry for
    entry. */
class HDGPostprocessBlocks
{
protected:
   const FiniteElementSpace *fes_q, *fes_p, *fes_s;
   Coefficient *ik{NULL};        ///< scalar 1/kappa, or NULL for the identity
   MatrixCoefficient *iK{NULL};  ///< matrix 1/kappa, or NULL
   int neq{0};
   int ir_order{-1};
   bool vector_flux{false};      ///< H(div) layout rather than scalar range

   /// Per-element offsets into the three data arrays below.
   Array<int> offs_B11, offs_s, offs_p;
   Array<real_t> B11_data;   ///< ns x na per element, DenseMatrix layout
   Array<real_t> c12_data;   ///< ns per element: B12 = c12 * mass_p^T
   Array<real_t> mass_data;  ///< nd per element: mass_p(j) = (phi_j, 1)_K

   bool assembled{false};
   long seq{-1};             ///< Mesh::sequence Assemble() saw
   long stamp{0};            ///< the caller's coefficient counter
   long stamp_seen{-1};      ///< its value at the last Assemble()

   /// Abort rather than hand back blocks that the mesh or a coefficient has
   /// outrun. Two integer comparisons, so it is affordable per element.
   void CheckFresh() const;

public:
   /** @param fes_q_ flux space; its layout is read from its range type, not
                     assumed -- a scalar-range space (L2, H1) carries
                     `neq*dim` components and a block is `dim` of them, an
                     H(div) space carries `neq` and a block is one.
       @param fes_p_ potential space; `neq` is taken from its vdim.
       @param fes_s_ the enriched space. The caller supplies it; see
                     HDGPotentialPostprocessor::Compute() for the default to
                     copy. Its NODE SET is a free choice and the method's
                     answer depends on it. */
   HDGPostprocessBlocks(const FiniteElementSpace &fes_q_,
                        const FiniteElementSpace &fes_p_,
                        const FiniteElementSpace &fes_s_);

   /** @brief Set the inverse diffusivity. Without one the identity is used.
       Setting either form clears the other. Neither is read after Assemble();
       announce a later move with CoefficientsMoved(). */
   void SetDiffusionInverse(Coefficient &c) { ik = &c; iK = NULL; }
   void SetDiffusionInverse(MatrixCoefficient &c) { iK = &c; ik = NULL; }

   /// Quadrature order; negative asks for one suited to the enriched space.
   void SetIntegrationOrder(int order) { ir_order = order; }

   /// Form and store every element's blocks. Idempotent.
   void Assemble();

   /// The Mesh::sequence and the coefficient stamp the last Assemble() saw.
   long GetSequence() const { return seq; }
   long GetCoefficientStamp() const { return stamp_seen; }

   /** @brief Announce that a diffusion coefficient has moved. The next
       Apply() or GetBlocks() aborts until Assemble() has been re-run. */
   void CoefficientsMoved() { stamp++; }

   /// True once Assemble() has run and no mesh or coefficient has outrun it.
   bool IsAssembled() const;

   const FiniteElementSpace &GetFluxSpace() const { return *fes_q; }
   const FiniteElementSpace &GetPotentialSpace() const { return *fes_p; }
   const FiniteElementSpace &GetEnrichedSpace() const { return *fes_s; }

   int GetNumEquations() const { return neq; }
   /// `ns` for element @a el, per equation.
   int NumNodes(int el) const { return offs_s[el+1] - offs_s[el]; }
   /// `na` for element @a el, per equation.
   int NumFluxDofs(int el) const;
   /// `nd` for element @a el, per equation.
   int NumPotentialDofs(int el) const { return offs_p[el+1] - offs_p[el]; }

   /** @brief Copies of element @a el's blocks, `B12` formed from its factors.

       B12 is rank one and is stored as those factors rather than as a matrix,
       so asking for it here costs an outer product. Apply() does not pay it. */
   void GetBlocks(int el, DenseMatrix &B11, DenseMatrix &B12) const;

   /** @brief `gamma` from an element state, per equation, allocation-free.

       @a u_l and @a p_l are element-local flux and potential vectors carrying
       every equation, and @a gamma comes back `neq*NumNodes(el)` long, all
       three laid out equation outermost.

       Reads no GridFunction, no ElementTransformation and no coefficient, so
       it is callable from inside a threaded element loop. */
   void Apply(int el, const Vector &u_l, const Vector &p_l,
              Vector &gamma) const;

   virtual ~HDGPostprocessBlocks() { }
};

/** @brief The classic HDG local postprocessing of the potential.

    On each element solve, for the potential of order `k+1`,

        (grad u*, grad v)_K = -(iK q_h, grad v)_K   for all v in P_{k+1}(K)
        (u*, 1)_K           =  (u_h, 1)_K

    which is Nguyen, Peraire & Cockburn eq (25) and gives `u*` converging at
    `k+2` where the theory offers it, one order better than `u_h`.

    The second equation is what closes the first. The local problem is a pure
    Neumann one -- its data is a flux and nothing fixes the constant -- so the
    element average of the computed potential is not an option among several
    but the definition, and it is applied unconditionally.

    **The `k+2` needs an O(1) stabilization, and this tree's default is not
    one.** HDGDiffusionIntegrator carries `1/h` intrinsically, which is the
    LDG-style scaling; an O(1) elementwise-constant `tau` is what the
    superconvergence theory uses and is reachable through the
    HDGStabilization hook. Measured on triangles, `-Delta u + u^3 - u = g`,
    `u* ` rates at k = 0,1,2,3: **1.00, 3.00, 4.01, 5.00** at O(1) `tau`
    against **-0.25, 1.96, 3.08** at the default. At k = 0 the default does not
    merely lose the extra order, the error stops decreasing and starts growing.
    So "the postprocessing did not superconverge" is a statement about `tau`
    before it is a statement about this class.

    **Systems.** Everything here is per equation. The flux, the potential and
    the result each carry `neq` blocks and the elements are solved one equation
    at a time, which is all a system needs of this postprocessing: the local
    problems do not couple, however strongly the equations couple globally.
    Blocks are laid out as the rest of the branch lays them out -- equation
    outermost -- and both flux layouts are read: a scalar-range space (L2, H1)
    with `vdim = neq*dim`, where block `e` is the component range
    `[e*dim, (e+1)*dim)`, and an H(div) space with `vdim = neq`, where block
    `e` is component `e`.

    **What this is not.** It postprocesses the potential only. The branch also
    carries a richer reconstruction -- DarcyForm::Reconstruct() -- which solves
    a mixed local problem for an enriched flux and traces as well, and is
    scalar-only. The two answer different questions and this one is deliberately
    the smaller: where the quantity wanted is a superconvergent potential per
    equation, it needs neither the trace space nor the hybridization, only the
    fields that have already been computed.

    **Compute() is HDGPostprocessBlocks::Apply() plus a scatter**, so there is
    exactly one copy of the element algebra. That changed the computed values
    in their last digits and it was not a defect: the loop this replaced
    accumulated the local right-hand side over quadrature points from the flux
    VALUES, and the blocks contract a matrix assembled once against the flux
    COEFFICIENTS. Same sum, different association -- and doing it in the second
    order is the entire reason the blocks exist, since a caller that evaluates
    this map every Newton step then pays no quadrature at all. Measured over
    orders 1-3, 2-D and 3-D, neq 1-3, both flux layouts and a non-diagonal
    x-dependent iK: 2.3e-15 to 3.5e-14 relative, none bit-identical. Do not
    write a test that pins these values against another implementation; pin
    them against a polynomial the method must reproduce exactly, which is what
    tests/unit/fem/test_darcy_postprocess.cpp does. */
class HDGPotentialPostprocessor
{
protected:
   const GridFunction *q;      ///< the computed flux, neq blocks
   const GridFunction *p;      ///< the computed potential, vdim = neq
   Coefficient *ik{NULL};      ///< scalar 1/kappa, or NULL for the identity
   MatrixCoefficient *iK{NULL};///< matrix 1/kappa, or NULL
   int neq{0};
   int ir_order{-1};

public:
   /** @brief Postprocess @a potential using @a flux.

       @a potential must have `vdim = neq`. @a flux must carry the same number
       of blocks in either layout described above; which one it is is read from
       its space rather than assumed. */
   HDGPotentialPostprocessor(const GridFunction &flux,
                             const GridFunction &potential);

   /// Number of equations, taken from the potential's vdim.
   int GetNumEquations() const { return neq; }

   /** @brief Set the inverse diffusivity, the `iK` above. Without one the
       identity is used, which is right when the flux is already the gradient
       up to sign. Setting either form clears the other. */
   void SetDiffusionInverse(Coefficient &c) { ik = &c; iK = NULL; }
   void SetDiffusionInverse(MatrixCoefficient &c) { iK = &c; ik = NULL; }

   /// Quadrature order; negative asks for one suited to the enriched space.
   void SetIntegrationOrder(int order) { ir_order = order; }

   /** @brief Compute the postprocessed potential into @a p_s.

       If @a p_s has no space, one is built for it and owned by it: the
       potential's collection at one higher order, with `vdim = neq`. */
   void Compute(GridFunction &p_s) const;

   virtual ~HDGPotentialPostprocessor() { }
};

}

#endif
