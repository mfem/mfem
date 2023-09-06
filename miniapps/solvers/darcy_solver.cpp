// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "darcy_solver.hpp"

using namespace std;
using namespace mfem;
using namespace blocksolvers;

namespace mfem
{
namespace blocksolvers
{
void SetOptions(IterativeSolver& solver, const IterSolveParameters& param)
{
   solver.SetPrintLevel(param.print_level);
   solver.SetMaxIter(param.max_iter);
   solver.SetAbsTol(param.abs_tol);
   solver.SetRelTol(param.rel_tol);
}

void DarcySolver::SetEliminatedSystems(std::shared_ptr<HypreParMatrix> M_e,
                                       std::shared_ptr<HypreParMatrix> B_e,
                                       const Array<int>& ess_tdof_list)
{
   M_e_ = M_e;
   B_e_ = B_e;
   rhs_needs_elimination_ = true;
   ess_tdof_list.Copy(ess_tdof_list_);
}

void DarcySolver::EliminateEssentialBC(const Vector &ess_data,
                                       Vector &rhs) const
{
   BlockVector blk_ess_data(ess_data.GetData(), offsets_);
   BlockVector blk_rhs(rhs, offsets_);
   M_e_->Mult(-1.0, blk_ess_data.GetBlock(0), 1.0, blk_rhs.GetBlock(0));
   B_e_->Mult(-1.0, blk_ess_data.GetBlock(0), 1.0, blk_rhs.GetBlock(1));
   for (int dof : ess_tdof_list_) { rhs[dof] = ess_data[dof]; }
}

/// Wrapper Block Diagonal Preconditioned MINRES (ex5p)
/** Wrapper for assembling the discrete Darcy problem (ex5p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
**/
BDPMinresSolver::BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                                 IterSolveParameters param)
   : DarcySolver(M.NumRows(), B.NumRows()), op_(offsets_), prec_(offsets_),
     BT_(B.Transpose()), solver_(M.GetComm())
{
   op_.SetBlock(0,0, &M);
   op_.SetBlock(0,1, BT_.As<HypreParMatrix>());
   op_.SetBlock(1,0, &B);

   Vector Md;
   M.GetDiag(Md);
   BT_.As<HypreParMatrix>()->InvScaleRows(Md);
   S_.Reset(ParMult(&B, BT_.As<HypreParMatrix>()));
   BT_.As<HypreParMatrix>()->ScaleRows(Md);

   prec_.SetDiagonalBlock(0, new HypreDiagScale(M));
   prec_.SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
   static_cast<HypreBoomerAMG&>(prec_.GetDiagonalBlock(1)).SetPrintLevel(0);
   prec_.owns_blocks = true;

   SetOptions(solver_, param);
   solver_.SetOperator(op_);
   solver_.SetPreconditioner(prec_);
}

void BDPMinresSolver::Mult(const Vector & x, Vector & y) const
{
   Vector x_e(x);
   if (rhs_needs_elimination_) { EliminateEssentialBC(y, x_e);}
   solver_.Mult(x_e, y);
   for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
}
} // namespace blocksolvers
} // namespace mfem
