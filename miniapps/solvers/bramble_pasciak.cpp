// Copyright (c) 2023-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//          ----------------------------------------------------------
//               Bramble-Pasciak preconditioning for Darcy problem
//          ----------------------------------------------------------
//
// Main idea is to precondition the block system
//                 Ax = [ M  B^T ] [u] = [f]
//                      [ B   0  ] [p] = [g]
//     where:
//        M = \int_\Omega (k u_h) \cdot v_h dx,
//        B = -\int_\Omega (div_h u_h) q_h dx,
//        f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
//        g = \int_\Omega g_exact q_h dx,
//        u_h, v_h \in R_h (Raviart-Thomas finite element space),
//        q_h \in W_h (piecewise discontinuous polynomials),
//        D: subset of the boundary where natural boundary condition is imposed.
// with a block transformation of the form X = AN - Id
//                  X = [ A*invQ - Id    0   ]
//                      [     B*invQ    -Id  ]
// where N is defined by
//                  N = [ invQ    0 ]
//                      [   0     0 ]
// and Q is constructed such that Q and M-Q are both s.p.d.
//
// The codes allows the user to provide such Q, or to construct it from the
// element matrices A_T. Moreover, the user can provide a block preconditioner
//                  P = [ M_1    0  ]
//                      [  0    M_2 ]
// Using the particular preconditioner H, defined as
//                  H = [ A - Q    0  ]
//                      [  0      M_2 ]
// (where M_1 = Q), enables a simplified version of a CG iteration (BPCG), as it avoids
// the direct application of invH and X.
//
// The code allows to use (P)CG with P or H, and BPCG.
#include "bramble_pasciak.hpp"

using namespace std;
using namespace mfem;
using namespace blocksolvers;

/// Bramble-Pasciak Solver
BramblePasciakSolver::BramblePasciakSolver(
   const std::shared_ptr<ParBilinearForm> &mVarf,
   const std::shared_ptr<ParMixedBilinearForm> &bVarf,
   const BPCGParameters &param)
   : DarcySolver(mVarf->ParFESpace()->GetTrueVSize(),
                 bVarf->TestFESpace()->GetTrueVSize())
{
   MFEM_ASSERT((param.q_scaling>=0.0) && (param.q_scaling<=1.0),
               "Invalid Q-scaling factor: param.q_scaling " << param.q_scaling );
   M_.reset(mVarf->ParallelAssemble());
   B_.reset(bVarf->ParallelAssemble());
   Q_.reset(ConstructMassPreconditioner(*mVarf, param.q_scaling));

   Init(*M_, *B_, *Q_, param);
}

BramblePasciakSolver::BramblePasciakSolver(
   HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
   Solver &M0, Solver &M1,
   const BPCGParameters &param)
   : DarcySolver(M.NumRows(), B.NumRows())
{
   Init(M, B, Q, M0, M1, param);
}

void BramblePasciakSolver::Init(
   HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
   const BPCGParameters &param)
{
   auto Bt = new TransposeOperator(&B);
   // invQ
   // TODO
   // This is not general enough. We are assuming Q is diag
   // Not using invQ ...
   // HypreParMatrix *invQ = new HypreParMatrix(Q);
   // Vector diagQ;
   // Q.GetDiag(diagQ);
   // *invQ = 1.0;
   // invQ->InvScaleRows(diagQ);

   Vector diagM;
   M.GetDiag(diagM);
   auto BT = B.Transpose();
   auto invDBt = new HypreParMatrix(*BT);
   invDBt->InvScaleRows(diagM);
   auto S = ParMult(&B, invDBt);
   auto M0 = new HypreDiagScale(Q);
   auto M1 = new HypreBoomerAMG(*S);
   // auto solver_M1 = new HypreBoomerAMG(*block11);
   M1->SetPrintLevel(0);

   use_bpcg = param.use_bpcg;

   if (use_bpcg)
   {
      oop_ = new BlockOperator(offsets_);
      oop_->owns_blocks = false;
      oop_->SetBlock(0, 0, &M);
      oop_->SetBlock(0, 1, Bt);
      oop_->SetBlock(1, 0, &B);

      // cpc_ unused in bpcg
      auto temp_cpc = new BlockDiagonalPreconditioner(offsets_);
      temp_cpc->owns_blocks = true;
      temp_cpc->SetDiagonalBlock(0, M0);
      temp_cpc->SetDiagonalBlock(1, M1);
      // tri(1,0) = B M0 = B invQ
      auto id_m = new IdentityOperator(M.NumRows());
      auto id_b = new IdentityOperator(B.NumRows());
      auto BinvM0 = new ProductOperator(&B, M0, false, false);
      // tri
      auto temp_tri = new BlockOperator(offsets_);
      temp_tri->owns_blocks = true;
      temp_tri->SetBlock(0, 0, id_m);
      temp_tri->SetBlock(1, 1, id_b, -1.0);
      temp_tri->SetBlock(1, 0, BinvM0);

      ppc_ = new ProductOperator(temp_cpc, temp_tri, true, true);

      ipc_ = new BlockOperator(offsets_);
      ipc_->owns_blocks = false;
      ipc_->SetDiagonalBlock(0, M0);

      // bpcg
      solver_.Reset(new BPCGSolver(M.GetComm()));
      SetOptions(*solver_.As<BPCGSolver>(), param);
      {
         solver_.As<BPCGSolver>()->SetOperator(*oop_);
         solver_.As<BPCGSolver>()->SetIncompletePreconditioner(*ipc_);
         solver_.As<BPCGSolver>()->SetParticularPreconditioner(*ppc_);
      }
   }
   else
   {
      // oop_ unused in cg
      auto temp_oop = new BlockOperator(offsets_);
      temp_oop->owns_blocks = false;
      temp_oop->SetBlock(0, 0, &M);
      temp_oop->SetBlock(0, 1, Bt);
      temp_oop->SetBlock(1, 0, &B);

      // ipc_ unused in cg
      auto temp_ipc = new BlockOperator(offsets_);
      temp_ipc->owns_blocks = false;
      temp_ipc->SetDiagonalBlock(0, M0);

      // temp_AN = temp_oop * temp_ipc
      auto temp_AN = new ProductOperator(temp_oop, temp_ipc, true, true);

      // Required for updating the RHS
      auto id = new IdentityOperator(M.NumRows()+B.NumRows());
      map_ = new AddOperator(temp_AN, 1.0, id, -1.0, true, true);

      mop_ = new ProductOperator(map_, temp_oop, false, true);

      cpc_ = new BlockDiagonalPreconditioner(offsets_);
      cpc_->owns_blocks = true;
      cpc_->SetDiagonalBlock(0, M0);
      cpc_->SetDiagonalBlock(1, M1);

      if (param.use_hpc)
      {
         auto Diff = new HypreParMatrix(M);
         Diff->Add(-1.0,Q);
         auto MM0 = new HypreDiagScale(*Diff);
         auto MM1 = new HypreDiagScale(*S);

         hpc_ = new BlockDiagonalPreconditioner(offsets_);
         hpc_->owns_blocks = true;
         hpc_->SetDiagonalBlock(0, MM0);
         hpc_->SetDiagonalBlock(1, MM1);
      }

      solver_.Reset(new CGSolver(M.GetComm()));
      SetOptions(*solver_.As<CGSolver>(), param);
      {
         solver_.As<CGSolver>()->SetOperator(*mop_);
         solver_.As<CGSolver>()->SetPreconditioner(*cpc_);
      }
   }
}

void BramblePasciakSolver::Init(
   HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
   Solver &M0, Solver &M1,
   const BPCGParameters &param)
{
   auto Bt = new TransposeOperator(&B);
   auto invQ = new HypreDiagScale(Q);

   use_bpcg = param.use_bpcg;

   if (use_bpcg)
   {
      oop_ = new BlockOperator(offsets_);
      oop_->owns_blocks = false;
      oop_->SetBlock(0, 0, &M);
      oop_->SetBlock(0, 1, Bt);
      oop_->SetBlock(1, 0, &B);

      // cpc_ unused in bpcg
      auto temp_cpc = new BlockDiagonalPreconditioner(offsets_);
      temp_cpc->owns_blocks = true;
      temp_cpc->SetDiagonalBlock(0, invQ);
      temp_cpc->SetDiagonalBlock(1, &M1);
      // tri(1,0) = B M0 = B invQ
      auto id_m = new IdentityOperator(M.NumRows());
      auto id_b = new IdentityOperator(B.NumRows());
      auto BinvQ = new ProductOperator(&B, invQ, false, false);
      // tri
      auto temp_tri = new BlockOperator(offsets_);
      temp_tri->owns_blocks = true;
      temp_tri->SetBlock(0, 0, id_m);
      temp_tri->SetBlock(1, 1, id_b, -1.0);
      temp_tri->SetBlock(1, 0, BinvQ);

      ppc_ = new ProductOperator(temp_cpc, temp_tri, true, true);

      ipc_ = new BlockOperator(offsets_);
      ipc_->owns_blocks = false;
      ipc_->SetDiagonalBlock(0, invQ);

      // bpcg
      solver_.Reset(new BPCGSolver(M.GetComm()));
      SetOptions(*solver_.As<BPCGSolver>(), param);
      {
         solver_.As<BPCGSolver>()->SetOperator(*oop_);
         solver_.As<BPCGSolver>()->SetIncompletePreconditioner(*ipc_);
         solver_.As<BPCGSolver>()->SetParticularPreconditioner(*ppc_);
      }
   }
   else
   {
      // oop_ unused in cg
      auto temp_oop = new BlockOperator(offsets_);
      temp_oop->owns_blocks = false;
      temp_oop->SetBlock(0, 0, &M);
      temp_oop->SetBlock(0, 1, Bt);
      temp_oop->SetBlock(1, 0, &B);

      // ipc_ unused in cg
      auto temp_ipc = new BlockOperator(offsets_);
      temp_ipc->owns_blocks = false;
      temp_ipc->SetDiagonalBlock(0, invQ);

      // temp_AN = temp_oop * temp_ipc
      auto temp_AN = new ProductOperator(temp_oop, temp_ipc, true, true);

      // Required for updating the RHS
      auto id = new IdentityOperator(M.NumRows()+B.NumRows());
      map_ = new AddOperator(temp_AN, 1.0, id, -1.0, true, true);

      mop_ = new ProductOperator(map_, temp_oop, false, true);

      cpc_ = new BlockDiagonalPreconditioner(offsets_);
      cpc_->owns_blocks = true;
      cpc_->SetDiagonalBlock(0, &M0);
      cpc_->SetDiagonalBlock(1, &M1);

      solver_.Reset(new CGSolver(M.GetComm()));
      SetOptions(*solver_.As<CGSolver>(), param);
      {
         solver_.As<CGSolver>()->SetOperator(*mop_);
         solver_.As<CGSolver>()->SetPreconditioner(*cpc_);
      }
   }
}

HypreParMatrix *BramblePasciakSolver::ConstructMassPreconditioner(
   ParBilinearForm &mVarf, double alpha)
{
   ParBilinearForm qVarf(mVarf.ParFESpace());
   for (int i = 0; i < mVarf.ParFESpace()->GetNE(); ++i)
   {
      DenseMatrix M_i, Q_i, evec;
      Vector eval, diag_i;
      double scaling = 0.0;

      mVarf.ComputeElementMatrix(i, M_i);
      M_i.GetDiag(diag_i);
      // M_i <- D^{-1/2} M_i D^{-1/2}, where D = diag(M_i)
      M_i.InvSymmetricScaling(diag_i);
      // M_i x = ev diag(M_i) x
      M_i.Eigenvalues(eval, evec);

      scaling = alpha*eval.Min();
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
      solver_.As<CGSolver>()->Mult(transformed_rhs, y);
   }
   else
   {
      solver_.As<BPCGSolver>()->Mult(x, y);
   }
   for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
}

int BramblePasciakSolver::GetNumIterations() const
{
   if (!use_bpcg) { return solver_.As<CGSolver>()->GetNumIterations(); }
   else { return solver_.As<BPCGSolver>()->GetNumIterations(); }
}

/// Bramble-Pasciak CG
void BPCGSolver::UpdateVectors()
{
   MemoryType mt = GetMemoryType(oper->GetMemoryClass());

   r.SetSize(width, mt); r.UseDevice(true);
   p.SetSize(width, mt); p.UseDevice(true);
   g.SetSize(width, mt); g.UseDevice(true);
   t.SetSize(width, mt); t.UseDevice(true);
   // r_hat.SetSize(width, mt); r_hat.UseDevice(true);
   r_bar.SetSize(width, mt); r_bar.UseDevice(true);
   r_red.SetSize(width, mt); r_red.UseDevice(true);
   g_red.SetSize(width, mt); g_red.UseDevice(true);
}

void BPCGSolver::Mult(const Vector &b, Vector &x) const
{
   int i;
   double delta, delta0, del0;
   double alpha, beta, gamma;

   // Initialization
   x.UseDevice(true);
   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
      // tra_->Mult(r,r_hat); // r_hat = X r
      // map_->Mult(r,r_tem); // r_tem = S r
      pprec->Mult(r,r_bar); // r_bar = P r
      p = r_bar;
      oper->Mult(p, g); // g = A p
      oper->Mult(r_bar, t); // t = A r_bar
      iprec->Mult(r, r_red); // r_red = N r
   }
   else
   {
      // TODO
      MFEM_ABORT("To implement non-iterative mode: iterative_mode: " <<
                 iterative_mode);
   }

   // Initial norms
   delta = delta0 = Dot(t, r_red) - Dot(r_bar, r); // Dot(r_bar, r_hat)
   if (delta0 >= 0.0) { initial_norm = sqrt(delta0); }
   MFEM_ASSERT(IsFinite(delta), "nom = " << delta);
   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
                << delta << (print_options.first_and_last ? " ...\n" : "\n");
   }
   Monitor(0, delta, r, x);

   if (delta < 0.0)
   {
      if (print_options.warnings)
      {
         mfem::out << "BPCG: The preconditioner is not positive definite. (Br, r) = "
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

   // MFEM checks some system properties before running the loop
   // Step 0.1: Compute (p,XAp), p = r_bar
   iprec->Mult(g, g_red);
   gamma = Dot(g, g_red) - Dot(g,p);
   MFEM_ASSERT(IsFinite(gamma), "den = " << gamma);
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
      // Step 2: Get new step in the search direction p
      alpha = delta0/gamma;
      // Step 3: Update solution (and residual) in the search direction
      add(x,  alpha, p, x);     // x = x + alpha p
      add(r, -alpha, g, r);     // r = r - alpha g
      // map_->Mult(r, r_tem);     // r_tem = S r
      pprec->Mult(r, r_bar); // r_bar = P r
      // Step 4: Compute (HXr,Xr) = (r_bar, r_hat)
      iprec->Mult(r, r_red);     // r_red = N r
      oper->Mult(r_bar, t);     // t = A r_bar
      delta = Dot(t, r_red) - Dot(r_bar,r);
      // Check
      MFEM_ASSERT(IsFinite(delta), "betanom = " << delta);
      if (delta < 0.0)
      {
         if (print_options.warnings)
         {
            mfem::out << "BPCG: The preconditioner is not positive definite. (Br, r) = "
                      << delta << '\n';
         }
         converged = false;
         final_iter = i;
         break;
      }
      if (print_options.iterations)
      {
         mfem::out << "   Iteration : " << setw(3) << i << "  (B r, r) = "
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
      // End checks
      // Step 5: Update search direction
      beta = delta/delta0;
      add(r_bar, beta, p, p);
      // Step 6: Update remaining directions
      // oper->Mult(r_bar, t);     // t = A r_bar
      add(t, beta, g, g);
      delta0 = delta;
      // Step 1: Compute (p,XAp)
      iprec->Mult(g, g_red);
      gamma = Dot(g, g_red) - Dot(g,p);
      MFEM_ASSERT(IsFinite(gamma), "den = " << gamma);
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
      mfem::out << "   Iteration : " << setw(3) << final_iter << "  (B r, r) = "
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
