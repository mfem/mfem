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

#include "bramble_pasciak.hpp"

using namespace std;

namespace mfem::blocksolvers
{

/// Bramble-Pasciak Solver
BramblePasciakSolver::BramblePasciakSolver(ParBilinearForm &mVarf,
                                           ParMixedBilinearForm &bVarf,
                                           const BPSParameters &param)
   : DarcySolver(mVarf.ParFESpace()->GetTrueVSize(),
                 bVarf.TestFESpace()->GetTrueVSize())
{
   M_.reset(mVarf.ParallelAssemble());
   B_.reset(bVarf.ParallelAssemble());
   Q_.reset(ConstructMassPreconditioner(mVarf, param.q_scaling));

   Vector diagM;
   M_->GetDiag(diagM);
   std::unique_ptr<HypreParMatrix> invDBt(B_->Transpose());
   invDBt->InvScaleRows(diagM);
   S_.reset(ParMult(B_.get(), invDBt.get(), true));
   M0_.Reset(new HypreDiagScale(*M_));
   M1_.Reset(new HypreBoomerAMG(*S_));
   M1_.As<HypreBoomerAMG>()->SetPrintLevel(0);

   Init(*M_, *B_, *Q_, *M0_.As<Solver>(), *M1_.As<Solver>(), param);
}

BramblePasciakSolver::BramblePasciakSolver(HypreParMatrix &M,
                                           HypreParMatrix &B,
                                           HypreParMatrix &Q,
                                           Solver &M0, Solver &M1,
                                           const BPSParameters &param)
   : DarcySolver(M.NumRows(), B.NumRows())
{
   Init(M, B, Q, M0, M1, param);
}

void BramblePasciakSolver::Init(HypreParMatrix &M,
                                HypreParMatrix &B,
                                HypreParMatrix &Q,
                                Solver &M0, Solver &M1,
                                const BPSParameters &param)
{
   Bt_ = std::make_unique<TransposeOperator>(&B);
   auto invQ = new HypreDiagScale(Q);

   use_bpcg = param.use_bpcg;

   if (use_bpcg)
   {
      oop_ = std::make_unique<BlockOperator>(offsets_);
      oop_->SetBlock(0, 0, &M);
      oop_->SetBlock(0, 1, Bt_.get());
      oop_->SetBlock(1, 0, &B);
      // cpc_ unused in bpcg
      auto temp_cpc = new BlockDiagonalPreconditioner(offsets_);
      temp_cpc->SetDiagonalBlock(0, invQ);
      temp_cpc->SetDiagonalBlock(1, &M1);
      // tri(1,0) = B M0 = B invQ
      auto id_m = new IdentityOperator(M.NumRows());
      auto id_b = new IdentityOperator(B.NumRows());
      auto BinvQ = new ProductOperator(&B, invQ, false, false);
      // tri
      auto temp_tri = new BlockOperator(offsets_);
      temp_tri->SetBlock(0, 0, id_m);
      temp_tri->SetBlock(1, 1, id_b, -1.0);
      temp_tri->SetBlock(1, 0, BinvQ);
      temp_tri->owns_blocks = 1;

      ppc_ = std::make_unique<ProductOperator>(temp_cpc, temp_tri, true, true);

      ipc_ = std::make_unique<BlockOperator>(offsets_);
      ipc_->SetDiagonalBlock(0, invQ);
      ipc_->owns_blocks = 1;

      // bpcg
      solver_ = std::make_unique<BPCGSolver>(M.GetComm(), ipc_.get(), ppc_.get());
      solver_->SetOperator(*oop_);
   }
   else
   {
      // oop_ unused in cg
      auto temp_oop = new BlockOperator(offsets_);
      temp_oop->SetBlock(0, 0, &M);
      temp_oop->SetBlock(0, 1, Bt_.get());
      temp_oop->SetBlock(1, 0, &B);

      // ipc_ unused in cg
      auto temp_ipc = new BlockOperator(offsets_);
      temp_ipc->SetDiagonalBlock(0, invQ);
      temp_ipc->owns_blocks = 1;

      // temp_AN = temp_oop * temp_ipc
      auto temp_AN = new ProductOperator(temp_oop, temp_ipc, true, true);

      // Required for updating the RHS
      auto id = new IdentityOperator(M.NumRows()+B.NumRows());
      map_ = std::make_unique<SumOperator>(temp_AN, 1.0, id, -1.0, true, true);
      mop_ = std::make_unique<ProductOperator>(map_.get(), temp_oop, false, false);

      cpc_ = std::make_unique<BlockDiagonalPreconditioner>(offsets_);
      cpc_->SetDiagonalBlock(0, &M0);
      cpc_->SetDiagonalBlock(1, &M1);

      // (P)CG
      solver_ = std::make_unique<CGSolver>(M.GetComm());
      solver_->SetOperator(*mop_);
      solver_->SetPreconditioner(*cpc_);
   }
   SetOptions(*solver_, param);
}

HypreParMatrix *BramblePasciakSolver::ConstructMassPreconditioner(
   const ParBilinearForm &mVarf, real_t q_scaling)
{
   MFEM_ASSERT((q_scaling > 0.0) && (q_scaling < 1.0),
               "Invalid Q-scaling factor: q_scaling = " << q_scaling );
   ParBilinearForm qVarf(mVarf.ParFESpace());
   qVarf.AllocateMatrix();
#ifndef MFEM_USE_LAPACK
   if (Mpi::Root())
   {
      mfem::out << "Warning: Using inverse power method to compute the minimum "
                << "eigenvalue of the small eigenvalue problem.\n";
      mfem::out << "         Consider compiling MFEM with LAPACK support.\n";
   }
#endif
   for (int i = 0; i < mVarf.ParFESpace()->GetNE(); ++i)
   {
      DenseMatrix M_i, Q_i;
      Vector diag_i;
      real_t scaling = 0.0, eval_i = 0.0;
      mVarf.ComputeElementMatrix(i, M_i);
      M_i.GetDiag(diag_i);
      // M_i <- D^{-1/2} M_i D^{-1/2}, where D = diag(M_i)
      M_i.InvSymmetricScaling(diag_i);
      // M_i x = ev diag(M_i) x
#ifdef MFEM_USE_LAPACK
      DenseMatrix evec;
      Vector eval;
      M_i.Eigenvalues(eval, evec);
      eval_i = eval.Min();
#else
      // Inverse power method
      Vector x(M_i.Height()), Mx(M_i.Height()), diff(M_i.Height());
      real_t eval_prev = 0.0;
      int iter = 0;
      x.Randomize(static_cast<int>(696383552LL+779345LL*i));
#if defined(MFEM_USE_DOUBLE)
      const real_t rel_tol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
      const real_t rel_tol = 1e-6;
#else
#error "Only single and double precision are supported!"
      const real_t rel_tol = 1e-12;
#endif
      DenseMatrixInverse M_i_inv(M_i);
      do
      {
         eval_prev = eval_i;
         M_i_inv.Mult(x, Mx);
         eval_i = Mx.Norml2();
         x.Set(1.0/eval_i, Mx);
         ++iter;
      }
      while ((iter < 1000) && (fabs(eval_i - eval_prev)/fabs(eval_i) > rel_tol));
      MFEM_VERIFY(fabs(eval_i - eval_prev)/fabs(eval_i) <= rel_tol,
                  "Inverse power method did not converge."
                  << "\n\t iter      = " << iter
                  << "\n\t eval_i    = " << eval_i
                  << "\n\t eval_prev = " << eval_prev
                  << "\n\t fabs(eval_i - eval_prev)/fabs(eval_i) = "
                  << fabs(eval_i - eval_prev)/fabs(eval_i));
      eval_i = 1.0/eval_i;
#endif
      scaling = q_scaling*eval_i;
      diag_i.Set(scaling, diag_i);
      Q_i.Diag(diag_i.GetData(), diag_i.Size());
      qVarf.AssembleElementMatrix(i, Q_i, 1);
   }
   qVarf.Finalize();
   return qVarf.ParallelAssemble();
}

void BramblePasciakSolver::Mult(const Vector & x, Vector & y) const
{
   if (!use_bpcg)
   {
      Vector transformed_rhs(x.Size());
      map_->Mult(x, transformed_rhs);
      solver_->Mult(transformed_rhs, y);
   }
   else
   {
      solver_->Mult(x, y);
   }
   for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
}

/// Bramble-Pasciak CG
void BPCGSolver::UpdateVectors()
{
   MemoryType mt = GetMemoryType(oper->GetMemoryClass());

   r.SetSize(width, mt); r.UseDevice(true);
   p.SetSize(width, mt); p.UseDevice(true);
   g.SetSize(width, mt); g.UseDevice(true);
   t.SetSize(width, mt); t.UseDevice(true);
   r_bar.SetSize(width, mt); r_bar.UseDevice(true);
   r_red.SetSize(width, mt); r_red.UseDevice(true);
   g_red.SetSize(width, mt); g_red.UseDevice(true);
}

void BPCGSolver::Mult(const Vector &b, Vector &x) const
{
   int i;
   real_t delta, delta0, del0;
   real_t alpha, beta, gamma;

   // Initialization
   x.UseDevice(true);
   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      r = b;
      x = 0.0;
   }

   pprec->Mult(r,r_bar);  // r_bar = P r
   p = r_bar;
   oper->Mult(p, g);      // g = A p
   oper->Mult(r_bar, t);  // t = A r_bar
   iprec->Mult(r, r_red); // r_red = N r

   delta = delta0 = Dot(t, r_red) - Dot(r_bar, r); // Dot(Pr, r)
   if (delta0 >= 0.0) { initial_norm = sqrt(delta0); }
   MFEM_ASSERT(IsFinite(delta), "norm = " << delta);
   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << 0 << "  (P r, r) = "
                << delta << (print_options.first_and_last ? " ...\n" : "\n");
   }
   Monitor(0, delta, r, x);

   if (delta < 0.0)
   {
      if (print_options.warnings)
      {
         mfem::out << "BPCG: The preconditioner is not positive definite. (Pr, r) = "
                   << delta << '\n';
      }
      converged = false;
      final_iter = 0;
      initial_norm = delta;
      final_norm = delta;
      return;
   }
   del0 = std::max(delta*rel_tol*rel_tol, abs_tol*abs_tol);
   if (delta <= del0)
   {
      converged = true;
      final_iter = 0;
      final_norm = sqrt(delta);
      return;
   }

   iprec->Mult(g, g_red);
   gamma = Dot(g, g_red) - Dot(g,p); // Dot(Ap, p)
   MFEM_ASSERT(IsFinite(gamma), "den (gamma) = " << gamma);
   if (gamma <= 0.0)
   {
      if (Dot(r_bar, r_bar) > 0.0 && print_options.warnings)
      {
         mfem::out << "BPCG: The operator is not positive definite. (Ar, r) = "
                   << gamma << '\n';
      }
      if (gamma == 0.0)
      {
         converged = false;
         final_iter = 0;
         final_norm = sqrt(delta);
         return;
      }
   }

   // Start iteration
   converged = false;
   final_iter = max_iter;
   for (i = 1; true; )
   {
      alpha = delta0/gamma;
      add(x,  alpha, p, x);  // x = x + alpha p
      add(r, -alpha, g, r);  // r = r - alpha g

      pprec->Mult(r, r_bar); // r_bar = P r
      iprec->Mult(r, r_red); // r_red = N r
      oper->Mult(r_bar, t);  // t = A r_bar
      delta = Dot(t, r_red) - Dot(r_bar,r);

      // Check
      MFEM_ASSERT(IsFinite(delta), "norm = " << delta);
      if (delta < 0.0)
      {
         if (print_options.warnings)
         {
            mfem::out << "BPCG: The preconditioner is not positive definite. (Pr, r) = "
                      << delta << '\n';
         }
         converged = false;
         final_iter = i;
         break;
      }
      if (print_options.iterations)
      {
         mfem::out << "   Iteration : " << setw(3) << i << "  (Pr, r) = "
                   << delta << std::endl;
      }
      Monitor(i, delta, r, x);
      if (delta <= del0)
      {
         converged = true;
         final_iter = i;
         break;
      }
      if (++i > max_iter)
      {
         break;
      }
      // End check

      beta = delta/delta0;
      add(r_bar, beta, p, p); // p = r_bar + beta p
      add(t, beta, g, g);     // g = t + beta g

      delta0 = delta;
      iprec->Mult(g, g_red);
      gamma = Dot(g, g_red) - Dot(g,p); // Dot(Ap, p)
      MFEM_ASSERT(IsFinite(gamma), "den (gamma) = " << gamma);
      if (gamma <= 0.0)
      {
         if (Dot(r_bar, r_bar) > 0.0 && print_options.warnings)
         {
            mfem::out << "BPCG: The operator is not positive definite. (Ar, r) = "
                      << gamma << '\n';
         }
         if (gamma == 0.0)
         {
            final_iter = i;
            break;
         }
      }
   }

   if (print_options.first_and_last && !print_options.iterations)
   {
      mfem::out << "   Iteration : " << setw(3) << final_iter << "  (Pr, r) = "
                << delta << '\n';
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "BPCG: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.summary || print_options.iterations ||
       print_options.first_and_last)
   {
      const auto arf = pow (gamma/delta0, 0.5/final_iter);
      mfem::out << "Average reduction factor = " << arf << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "BPCG: No convergence!" << '\n';
   }

   final_norm = sqrt(delta);
   Monitor(final_iter, final_norm, r, x, true);
}

} // namespace mfem::blocksolvers
