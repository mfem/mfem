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

#include "darcy_solver.hpp"

using namespace std;

namespace mfem::blocksolvers
{

void SetOptions(IterativeSolver& solver, const IterSolveParameters& param)
{
   solver.SetPrintLevel(param.print_level);
   solver.SetMaxIter(param.max_iter);
   solver.SetAbsTol(param.abs_tol);
   solver.SetRelTol(param.rel_tol);
}

/// Wrapper Block Diagonal Preconditioned MINRES (ex5p)
/** Wrapper for assembling the discrete Darcy problem (ex5p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
*/
BDPMinresSolver::BDPMinresSolver(const HypreParMatrix& M,
                                 const HypreParMatrix& B,
                                 IterSolveParameters param)
   : DarcySolver(M.NumRows(), B.NumRows()), op_(offsets_), prec_(offsets_),
     BT_(B.Transpose()), solver_(M.GetComm())
{
   op_.SetBlock(0,0, const_cast<HypreParMatrix*>(&M));
   op_.SetBlock(0,1, BT_.As<HypreParMatrix>());
   op_.SetBlock(1,0, const_cast<HypreParMatrix*>(&B));

   Vector Md;
   M.GetDiag(Md);
   BT_.As<HypreParMatrix>()->InvScaleRows(Md);
   S_.Reset(ParMult(&B, BT_.As<HypreParMatrix>()));
   BT_.As<HypreParMatrix>()->ScaleRows(Md);

   prec_.SetDiagonalBlock(0, new HypreDiagScale(M));
   prec_.SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
   static_cast<HypreBoomerAMG&>(prec_.GetDiagonalBlock(1)).SetPrintLevel(0);
   prec_.owns_blocks = 1;

   SetOptions(solver_, param);
   solver_.SetOperator(op_);
   solver_.SetPreconditioner(prec_);
}

void BDPMinresSolver::Mult(const Vector & x, Vector & y) const
{
   solver_.Mult(x, y);
   for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
}

} // namespace mfem::blocksolvers
