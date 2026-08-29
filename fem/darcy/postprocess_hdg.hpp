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

namespace mfem
{

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
    fields that have already been computed. */
class HDGPotentialPostprocessor
{
protected:
   const GridFunction *q;      ///< the computed flux, neq blocks
   const GridFunction *p;      ///< the computed potential, vdim = neq
   Coefficient *ik{NULL};      ///< scalar 1/kappa, or NULL for the identity
   MatrixCoefficient *iK{NULL};///< matrix 1/kappa, or NULL
   int neq{0};
   int ir_order{-1};

   /// Value of block @a e of the flux at the current integration point.
   void GetFluxBlock(const FiniteElement &fe, ElementTransformation &T,
                     const IntegrationPoint &ip, const Vector &loc_q, int e,
                     Vector &q_e) const;

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
