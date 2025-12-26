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

#include "braginskii_hdg_solver.hpp"

namespace mfem
{
namespace plasma
{
ConvectionDiffusionOp::ConvectionDiffusionOp(const FiniteElementSpace &fes_p_,
                                             hdg::DarcyOperator &darcy_op_, Operator &conv_op_)
   : TimeDependentOperator(darcy_op_.Width(), 0., IMPLICIT),
     fes_p(fes_p_), darcy_op(darcy_op_), conv_op(conv_op_)
{
   const FiniteElementSpace *vfes_p = darcy_op.GetDarcyForm().PotentialFESpace();
   Minv.reset(new DGMassInverse(const_cast<FiniteElementSpace&>(fes_p)));
   bp.SetSize(vfes_p->GetVSize());
}

void ConvectionDiffusionOp::AddMult(const Vector &x, Vector &y, real_t a) const
{
   if (eval_mode == EvalMode::ADDITIVE_TERM_1) { return; }

   const Array<int> &offsets = darcy_op.GetOffsets();
   const BlockVector bx(const_cast<Vector&>(x), offsets);
   BlockVector by(y, offsets);

   conv_op.Mult(bx.GetBlock(1), bp);

   const FiniteElementSpace *vfes_p = darcy_op.GetDarcyForm().PotentialFESpace();
   const int ndofs = vfes_p->GetNDofs();
   const int vdim = vfes_p->GetVDim();
   for (int d = 0; d < vdim; d++)
   {
      const Vector bp_d(bp, d*ndofs, ndofs);
      Vector dp_d(by.GetBlock(1), d*ndofs, ndofs);
      Minv->AddMult(bp_d, dp_d, a);
   }
}

void ConvectionDiffusionOp::ImplicitSolve(const real_t dt, const Vector &x,
                                          Vector &y)
{
   if (eval_mode == EvalMode::NORMAL || eval_mode == EvalMode::ADDITIVE_TERM_1)
   {
      darcy_op.ImplicitSolve(dt, x, y);
   }

   if (eval_mode == EvalMode::NORMAL || eval_mode == EvalMode::ADDITIVE_TERM_2)
   {
      AddMult(x, y, -1.);
   }
}

} // namespace plasma
} // namespace mfem
