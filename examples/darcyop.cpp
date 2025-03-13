// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "darcyop.hpp"
#include "../general/tic_toc.hpp"
#include <fstream>

namespace mfem
{

void mfem::DarcyOperator::SetupNonlinearSolver(real_t rtol, real_t atol,
                                               int iters)
{
   IterativeSolver *lin_solver = NULL;
   switch (solver_type)
   {
      case SolverType::Default:
      case SolverType::LBFGS:
         prec = NULL;
#ifdef MFEM_USE_MPI
         solver = new LBFGSSolver(MPI_COMM_WORLD);
#else
         solver = new LBFGSSolver();
#endif
         solver_str = "LBFGS";
         break;
      case SolverType::LBB:
         prec = NULL;
#ifdef MFEM_USE_MPI
         solver = new LBBSolver(MPI_COMM_WORLD);
#else
         solver = new LBBSolver();
#endif
         solver_str = "LBB";
         break;
      case SolverType::Newton:
#ifdef MFEM_USE_MPI
         lin_solver = new GMRESSolver(MPI_COMM_WORLD);
#else
         lin_solver = new GMRESSolver();
#endif
         prec_str = "GMRES";
#ifdef MFEM_USE_MPI
         solver = new NewtonSolver(MPI_COMM_WORLD);
#else
         solver = new NewtonSolver();
#endif
         solver_str = "Newton";
         break;
      case SolverType::KINSol:
#ifdef MFEM_USE_SUNDIALS
#ifdef MFEM_USE_MPI
         lin_solver = new GMRESSolver(MPI_COMM_WORLD);
#else
         lin_solver = new GMRESSolver();
#endif
         prec_str = "GMRES";
#ifdef MFEM_USE_MPI
         solver = new KINSolver(MPI_COMM_WORLD, KIN_PICARD);
#else
         solver = new KINSolver(KIN_PICARD);
#endif
         static_cast<KINSolver*>(solver)->EnableAndersonAcc(10);
         solver_str = "KINSol";
#else
         MFEM_ABORT("Sundials not installed!");
#endif
         break;
   }

   if (lin_solver)
   {
      if (!darcy->GetHybridization())
      {
#ifdef MFEM_USE_MPI
         if (pdarcy)
         {
            lin_prec = new SchurPreconditioner(pdarcy, true);
         }
         else
#endif
            lin_prec = new SchurPreconditioner(darcy, true);
         lin_solver->SetPreconditioner(*lin_prec);
         prec_str += "+";
         prec_str += static_cast<SchurPreconditioner*>(lin_prec)->GetString();
      }

      lin_solver->SetAbsTol(atol);
      lin_solver->SetRelTol(rtol * 1e-2);
      lin_solver->SetMaxIter(iters);
      lin_solver->SetPrintLevel(0);
      prec = lin_solver;
   }

   solver->SetAbsTol(atol);
   solver->SetRelTol(rtol);
   solver->SetMaxIter(iters);
   if (prec) { solver->SetPreconditioner(*prec); }
   solver->SetPrintLevel((btime_u || btime_p)?0:1);
   solver->iterative_mode = true;
}

void DarcyOperator::SetupLinearSolver(real_t rtol, real_t atol, int iters)
{
   if (darcy->GetHybridization())
   {
#ifdef MFEM_USE_MPI
      prec = new HypreBoomerAMG();
      prec_str = "HypreAMG";
#else
      prec = new GSSmoother();
      prec_str = "GS";
#endif
   }
   else if (darcy->GetReduction())
   {
#ifdef MFEM_USE_MPI
      prec = new HypreBoomerAMG();
      prec_str = "HypreAMG";
#else
#ifndef MFEM_USE_SUITESPARSE
      prec = new GSSmoother();
      prec_str = "GS";
#else
      prec = new UMFPackSolver();
      prec_str = "UMFPack";
#endif
#endif
   }
   else
   {
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         prec = new SchurPreconditioner(pdarcy);
      }
      else
#endif
         prec = new SchurPreconditioner(darcy);
      prec_str = static_cast<SchurPreconditioner*>(prec)->GetString();
   }

#ifdef MFEM_USE_MPI
   solver = new GMRESSolver(MPI_COMM_WORLD);
#else
   solver = new GMRESSolver();
#endif
   solver_str = "GMRES";
   solver->SetAbsTol(atol);
   solver->SetRelTol(rtol);
   solver->SetMaxIter(iters);
   if (prec) { solver->SetPreconditioner(*prec); }
   solver->SetPrintLevel((btime_u || btime_p)?0:1);
   solver->iterative_mode = true;
}

DarcyOperator::DarcyOperator(const Array<int> &ess_flux_tdofs_list_,
                             DarcyForm *darcy_, LinearForm *g_, LinearForm *f_, LinearForm *h_,
                             const Array<Coefficient*> &coeffs_, SolverType stype_, bool btime_u_,
                             bool btime_p_)
   : TimeDependentOperator(0, 0., IMPLICIT),
     ess_flux_tdofs_list(ess_flux_tdofs_list_), darcy(darcy_), g(g_), f(f_), h(h_),
     coeffs(coeffs_), solver_type(stype_), btime_u(btime_u_), btime_p(btime_p_)
{
   offsets = ConstructOffsets(*darcy);
   width = height = offsets.Last();

   if (darcy->GetHybridization())
   {
      trace_space = darcy->GetHybridization()->ConstraintFESpace();
   }

   if (btime_u || btime_p)
      idtcoeff = new FunctionCoefficient([&](const Vector &) { return idt; });

   if (btime_u)
   {
      BilinearForm *Mq = const_cast<BilinearForm*>(
                            (const_cast<const DarcyForm*>(darcy))->GetFluxMassForm());
      NonlinearForm *Mqnl = const_cast<NonlinearForm*>(
                               (const_cast<const DarcyForm*>(darcy))->GetFluxMassNonlinearForm());
      const int dim = darcy->FluxFESpace()->GetMesh()->Dimension();
      const bool dg = (darcy->FluxFESpace()->FEColl()->GetRangeType(
                          dim) == FiniteElement::SCALAR);
      if (Mq)
      {
         if (dg)
         {
            Mq->AddDomainIntegrator(new VectorMassIntegrator(*idtcoeff));
         }
         else
         {
            Mq->AddDomainIntegrator(new VectorFEMassIntegrator(*idtcoeff));
         }
      }
      if (Mqnl)
      {
         if (dg)
         {
            Mqnl->AddDomainIntegrator(new VectorMassIntegrator(*idtcoeff));
         }
         else
         {
            Mqnl->AddDomainIntegrator(new VectorFEMassIntegrator(*idtcoeff));
         }

         if (trace_space)
         {
            //hybridization must be reconstructed, since the non-linear
            //potential mass must be passed to it
            darcy->EnableHybridization(trace_space,
                                       new NormalTraceJumpIntegrator(),
                                       ess_flux_tdofs_list);
         }
      }
      Mq0 = new BilinearForm(darcy->FluxFESpace());
      if (dg)
      {
         Mq0->AddDomainIntegrator(new VectorMassIntegrator(*idtcoeff));
      }
      else
      {
         Mq0->AddDomainIntegrator(new VectorFEMassIntegrator(*idtcoeff));
      }
   }

   if (btime_p)
   {
      BilinearForm *Mt = const_cast<BilinearForm*>(
                            (const_cast<const DarcyForm*>(darcy))->GetPotentialMassForm());
      NonlinearForm *Mtnl = const_cast<NonlinearForm*>(
                               (const_cast<const DarcyForm*>(darcy))->GetPotentialMassNonlinearForm());
      if (Mt) { Mt->AddDomainIntegrator(new MassIntegrator(*idtcoeff)); }
      if (Mtnl)
      {
         Mtnl->AddDomainIntegrator(new MassIntegrator(*idtcoeff));
         if (trace_space)
         {
            //hybridization must be reconstructed, since the non-linear
            //potential mass must be passed to it
            darcy->EnableHybridization(trace_space,
                                       new NormalTraceJumpIntegrator(),
                                       ess_flux_tdofs_list);
         }
      }
      Mt0 = new BilinearForm(darcy->PotentialFESpace());
      Mt0->AddDomainIntegrator(new MassIntegrator(*idtcoeff));
   }
}

#ifdef MFEM_USE_MPI
DarcyOperator::DarcyOperator(const Array<int> &ess_flux_tdofs_list,
                             ParDarcyForm *darcy_, ParLinearForm *g_, ParLinearForm *f_, ParLinearForm *h_,
                             const Array<Coefficient *> &coeffs, SolverType stype, bool bflux_u,
                             bool btime_p)
   : DarcyOperator(ess_flux_tdofs_list, (DarcyForm*) darcy_, g_, f_, h_, coeffs,
                   stype, bflux_u, btime_p)
{
   pdarcy = darcy_;
   pg = g_;
   pf = f_;
   ph = h_;
}
#endif //MFEM_USE_MPI

DarcyOperator::~DarcyOperator()
{
   delete lin_prec;
   delete prec;
   delete solver;
   delete monitor;
   delete Mt0;
   delete Mq0;
   delete idtcoeff;
}

Array<int> DarcyOperator::ConstructOffsets(const DarcyForm &darcy)
{
   if (!darcy.GetHybridization())
   {
      return darcy.GetOffsets();
   }

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = darcy.FluxFESpace()->GetVSize();
   offsets[2] = darcy.PotentialFESpace()->GetVSize();
   offsets[3] = darcy.GetHybridization()->ConstraintFESpace()->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void DarcyOperator::ImplicitSolve(const real_t dt, const Vector &x_v,
                                  Vector &dx_v)
{
#ifdef MFEM_USE_MPI
   const bool verbose = Mpi::Root();
#else
   const bool verbose = true;
#endif

   //form the linear system

   BlockVector rhs(g->GetData(), darcy->GetOffsets());
   BlockVector x(dx_v.GetData(), darcy->GetOffsets());
   dx_v = x_v;

   //set time

   for (Coefficient *coeff : coeffs)
   {
      coeff->SetTime(t);
   }

   //assemble rhs

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

#ifdef MFEM_USE_MPI
   if (pdarcy)
   {
      pg->Assemble();
      pf->Assemble();
      if (ph) { ph->Assemble(); }
   }
   else
#endif //MFEM_USE_MPI
   {
      g->Assemble();
      f->Assemble();
      if (h) { h->Assemble(); }
   }

   //check if the operator has to be reassembled

   bool reassemble = (idt != 1./dt);

   if (reassemble)
   {
      idt = 1./dt;

      //reset the operator

      darcy->Update();

      //assemble the system
#ifdef MFEM_USE_MPI
      if (pdarcy)
      {
         pdarcy->Assemble();
      }
      else
#endif //MFEM_USE_MPI
      {
         darcy->Assemble();
      }

      if (Mq0)
      {
         Mq0->Update();
         Mq0->Assemble();
         //Mq0->Finalize();
      }
      if (Mt0)
      {
         Mt0->Update();
         Mt0->Assemble();
         //Mt0->Finalize();
      }
   }

   if (Mq0)
   {
      GridFunction u_h;
      u_h.MakeRef(darcy->FluxFESpace(), x.GetBlock(0), 0);
      Mq0->AddMult(u_h, *g, +1.);
   }

   if (Mt0)
   {
      GridFunction p_h;
      p_h.MakeRef(darcy->PotentialFESpace(), x.GetBlock(1), 0);
      Mt0->AddMult(p_h, *f, -1.);
   }
#if 0
   if (Mq0 && Mt0)
   {
      GridFunction u_h, p_h;
      u_h.MakeRef(darcy->FluxFESpace(), x.GetBlock(0), 0);
      p_h.MakeRef(darcy->PotentialFESpace(), x.GetBlock(1), 0);
      darcy->GetFluxDivForm()->AddMultTranspose(p_h, *g, -1.);
      darcy->GetFluxDivForm()->AddMult(u_h, *f, +1.);
   }
#endif
   //form the reduced system

   OperatorHandle op;
   Vector X, RHS;
   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (ph)
      {
         RHS.SetSize(trace_space->GetTrueVSize());
         ph->ParallelAssemble(RHS);
      }
#else
      X.MakeRef(dx_v, offsets[2], trace_space->GetVSize());
      RHS.MakeRef(*h, 0, trace_space->GetVSize());
#endif
   }

   darcy->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                           op, X, RHS);


   chrono.Stop();
   if (verbose) { std::cout << "Assembly took " << chrono.RealTime() << "s.\n"; }

   if (reassemble)
   {
      // 10. Construct the preconditioner and solver

      chrono.Clear();
      chrono.Start();

      constexpr int maxIter(1000);
      constexpr real_t rtol(1.e-6);
      constexpr real_t atol(1.e-10);

      // We do not want to initialize any new forms here, only obtain
      // the existing ones, so we const cast the DarcyForm
      const DarcyForm *cdarcy = const_cast<const DarcyForm*>(darcy);

      //const BilinearForm *Mq = cdarcy->GetFluxMassForm();
      const NonlinearForm *Mqnl = cdarcy->GetFluxMassNonlinearForm();
      const BlockNonlinearForm *Mnl = cdarcy->GetBlockNonlinearForm();
      //const MixedBilinearForm *B = cdarcy->GetFluxDivForm();
      //const BilinearForm *Mt = cdarcy->GetPotentialMassForm();
      const NonlinearForm *Mtnl = cdarcy->GetPotentialMassNonlinearForm();

      if (trace_space) //hybridization
      {
         if (Mqnl || Mtnl || Mnl)
         {
            darcy->GetHybridization()->SetLocalNLSolver(
               DarcyHybridization::LSsolveType::Newton,
               maxIter, rtol * 1e-3, atol, -1);
            lsolver_str = "Newton+GMRES";

            SetupNonlinearSolver(rtol, atol, maxIter);
         }
         else
         {
            SetupLinearSolver(rtol, atol, maxIter);

            if (monitor_step >= 0)
            {
               monitor = new IterativeGLVis(this, monitor_step);
               solver->SetMonitor(*monitor);
            }
         }

         solver->SetOperator(*op);
      }
      else if (darcy->GetReduction()) //reduction
      {
         SetupLinearSolver(rtol, atol, maxIter);

         solver->SetOperator(*op);
      }
      else //mixed
      {
         if ((Mqnl || Mtnl || Mnl) && solver_type != SolverType::Default)
         {
            SetupNonlinearSolver(rtol, atol, maxIter);

            if (prec)
            {
               if (ess_flux_tdofs_list.Size() > 0)
               {
                  MFEM_ABORT("Gradient is not implemented with essential DOFs!");
               }
               solver->SetOperator(*darcy);
            }
            else
            {
               solver->SetOperator(*op);
            }
         }
         else
         {
            if (Mqnl || Mtnl || Mnl)
            {
               std::cerr << "A linear solver is used for a non-linear problem!" << std::endl;
            }

            SetupLinearSolver(rtol, atol, maxIter);

            solver->SetOperator(*op);
         }
      }

      chrono.Stop();
      if (verbose) { std::cout << "Preconditioner took " << chrono.RealTime() << "s.\n"; }
   }

   // 11. Solve the linear system with GMRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();

   solver->Mult(RHS, X);

#ifdef MFEM_USE_MPI
   if (pdarcy)
   {
      pdarcy->RecoverFEMSolution(X, rhs, x);
   }
   else
#endif
   {
      darcy->RecoverFEMSolution(X, rhs, x);
   }

#ifdef MFEM_USE_MPI
   if (trace_space)
   {
      Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
      trace_space->GetProlongationMatrix()->Mult(X, x_r);
   }
#endif

   chrono.Stop();

   if (verbose)
   {
      std::cout << solver_str;
      if (!prec_str.empty()) { std::cout << "+" << prec_str; }
      if (!lsolver_str.empty()) { std::cout << "/" << lsolver_str; }
      if (solver->GetConverged())
      {
         std::cout << " converged in " << solver->GetNumIterations()
                   << " iterations with a residual norm of " << solver->GetFinalNorm()
                   << ".\n";
      }
      else
      {
         std::cout << " did not converge in " << solver->GetNumIterations()
                   << " iterations. Residual norm is " << solver->GetFinalNorm()
                   << ".\n";
      }
      std::cout << "solver took " << chrono.RealTime() << "s.\n";
   }

   dx_v -= x_v;
   dx_v *= idt;
}


DarcyOperator::SchurPreconditioner::SchurPreconditioner(const DarcyForm *darcy_,
                                                        bool nonlinear_)
   : Solver(darcy_->Height()), darcy(darcy_), nonlinear(nonlinear_)
{
   if (!nonlinear)
   {
      Vector x(Width());
      x = 0.;
      Construct(x);
   }

#ifndef MFEM_USE_SUITESPARSE
   prec_str = "GS";
#else
   prec_str = "UMFPack";
#endif
}

#ifdef MFEM_USE_MPI
DarcyOperator::SchurPreconditioner::SchurPreconditioner(
   const ParDarcyForm *darcy_, bool nonlinear_)
   : Solver(darcy_->Height()), darcy(darcy_), pdarcy(darcy_), nonlinear(nonlinear_)
{
   if (!nonlinear)
   {
      Vector x(Width());
      x = 0.;
      ConstructPar(x);
   }

   prec_str = "HypreAMG";
}
#endif //MFEM_USE_MPI

DarcyOperator::SchurPreconditioner::~SchurPreconditioner()
{
   delete darcyPrec;
   delete S;
}

void DarcyOperator::SchurPreconditioner::Construct(const Vector &x_v) const

{
   const Array<int> &block_offsets = darcy->GetOffsets();
   BlockVector x(x_v.GetData(), block_offsets);

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (darcy->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const BilinearForm *Mq = darcy->GetFluxMassForm();
   const NonlinearForm *Mqnl = darcy->GetFluxMassNonlinearForm();
   const BlockNonlinearForm *Mnl = darcy->GetBlockNonlinearForm();
   const MixedBilinearForm *B = darcy->GetFluxDivForm();
   const BilinearForm *Mt = darcy->GetPotentialMassForm();
   const NonlinearForm *Mtnl = darcy->GetPotentialMassNonlinearForm();

   Vector Md(block_offsets[1] - block_offsets[0]);
   delete darcyPrec;
   darcyPrec = new BlockDiagonalPreconditioner(block_offsets);
   darcyPrec->owns_blocks = true;
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Mq->Height());
      for (int i=0; i<Mq->Height(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(B->Height());
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
   }
   else
   {
      BlockOperator *bop = NULL;

      // get diagonal
      if (Mq)
      {
         const SparseMatrix &Mqm(Mq->SpMat());
         Mqm.GetDiag(Md);
         invM = new DSmoother(Mqm);
      }
      else if (Mqnl)
      {
         const SparseMatrix &Mqm = static_cast<SparseMatrix&>(
                                      Mqnl->GetGradient(x.GetBlock(0)));
         Mqm.GetDiag(Md);
         invM = new DSmoother(Mqm);
      }
      else if (Mnl)
      {
         bop = static_cast<BlockOperator*>(&Mnl->GetGradient(x));

         const SparseMatrix &Mqm = static_cast<SparseMatrix&>(bop->GetBlock(0,0));

         Mqm.GetDiag(Md);
         invM = new DSmoother(Mqm);
      }

      Md.HostReadWrite();

      const SparseMatrix &Bm(B->SpMat());
      SparseMatrix *MinvBt = Transpose(Bm);

      for (int i = 0; i < Md.Size(); i++)
      {
         MinvBt->ScaleRow(i, 1./Md(i));
      }

      delete S;
      S = mfem::Mult(Bm, *MinvBt);
      delete MinvBt;

      if (Mt)
      {
         const SparseMatrix &Mtm(Mt->SpMat());
         SparseMatrix *Snew = Add(Mtm, *S);
         delete S;
         S = Snew;
      }
      else if (Mtnl)
      {
         const SparseMatrix &Mtm = static_cast<SparseMatrix&>(
                                      Mtnl->GetGradient(x.GetBlock(1)));
         SparseMatrix *Snew = Add(Mtm, *S);
         delete S;
         S = Snew;
      }
      if (Mnl)
      {
         const SparseMatrix &Mtm = static_cast<SparseMatrix&>(bop->GetBlock(1,1));
         if (Mtm.NumNonZeroElems() > 0)
         {
            SparseMatrix *Snew = Add(Mtm, *S);
            delete S;
            S = Snew;
         }
      }

#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec->SetDiagonalBlock(0, invM);
   darcyPrec->SetDiagonalBlock(1, invS);
}

#ifdef MFEM_USE_MPI
void DarcyOperator::SchurPreconditioner::ConstructPar(const Vector &x_v) const
{
   const Array<int> &block_offsets = pdarcy->GetTrueOffsets();
   BlockVector x(x_v.GetData(), block_offsets);

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (darcy->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const ParBilinearForm *Mq = pdarcy->GetParFluxMassForm();
   //const NonlinearForm *Mqnl = darcy->GetFluxMassNonlinearForm();
   //const BlockNonlinearForm *Mnl = darcy->GetBlockNonlinearForm();
   const ParMixedBilinearForm *B = pdarcy->GetParFluxDivForm();
   const ParBilinearForm *Mt = pdarcy->GetParPotentialMassForm();
   //const NonlinearForm *Mtnl = darcy->GetPotentialMassNonlinearForm();

   Vector Md(block_offsets[1] - block_offsets[0]);
   delete darcyPrec;
   darcyPrec = new BlockDiagonalPreconditioner(block_offsets);
   darcyPrec->owns_blocks = true;
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Mq->Height());
      for (int i=0; i<Mq->Height(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(B->Height());
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
   }
   else
   {
      //BlockOperator *bop = NULL;

      // get diagonal
      if (Mq)
      {
         const HypreParMatrix *Mqm = const_cast<ParBilinearForm*>
                                     (Mq)->ParallelAssembleInternal();
         Mqm->GetDiag(Md);
         invM = new HypreDiagScale(*Mqm);
      }
      /*else if (Mqnl)
      {
         const SparseMatrix &Mqm = static_cast<SparseMatrix&>(
                                      Mqnl->GetGradient(x.GetBlock(0)));
         Mqm.GetDiag(Md);
         invM = new DSmoother(Mqm);
      }
      else if (Mnl)
      {
         bop = static_cast<BlockOperator*>(&Mnl->GetGradient(x));

         const SparseMatrix &Mqm = static_cast<SparseMatrix&>(bop->GetBlock(0,0));

         Mqm.GetDiag(Md);
         invM = new DSmoother(Mqm);
      }*/

      Md.HostReadWrite();

      const HypreParMatrix *Bm = const_cast<ParMixedBilinearForm*>
                                 (B)->ParallelAssembleInternal();
      HypreParMatrix *MinvBt = Bm->Transpose();
      MinvBt->InvScaleRows(Md);

      delete hS;
      hS = mfem::ParMult(Bm, MinvBt, true);
      delete MinvBt;

      if (Mt)
      {
         const HypreParMatrix *Mtm = const_cast<ParBilinearForm*>
                                     (Mt)->ParallelAssembleInternal();
         HypreParMatrix *hSnew = ParAdd(Mtm, hS);
         delete hS;
         hS = hSnew;
      }
      /*else if (Mtnl)
      {
         const SparseMatrix &Mtm = static_cast<SparseMatrix&>(
                                      Mtnl->GetGradient(x.GetBlock(1)));
         SparseMatrix *Snew = Add(Mtm, *S);
         delete S;
         S = Snew;
      }
      if (Mnl)
      {
         const SparseMatrix &Mtm = static_cast<SparseMatrix&>(bop->GetBlock(1,1));
         if (Mtm.NumNonZeroElems() > 0)
         {
            SparseMatrix *Snew = Add(Mtm, *S);
            delete S;
            S = Snew;
         }
      }*/

      invS = new HypreBoomerAMG(*hS);
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   darcyPrec->SetDiagonalBlock(0, invM);
   darcyPrec->SetDiagonalBlock(1, invS);
}
#endif

DarcyOperator::IterativeGLVis::IterativeGLVis(DarcyOperator *p_, int step_)
   : p(p_), step(step_)
{
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   q_sock.open(vishost, visport);
   q_sock.precision(8);
   t_sock.open(vishost, visport);
   t_sock.precision(8);
}

void DarcyOperator::IterativeGLVis::MonitorSolution(int it, real_t norm,
                                                    const Vector &X, bool final)
{
   if (step != 0 && it % step != 0 && !final) { return; }

   BlockVector x(p->darcy->GetOffsets()); x = 0.;
   BlockVector rhs(p->g->GetData(), p->darcy->GetOffsets());
   p->darcy->RecoverFEMSolution(X, rhs, x);

   GridFunction q_h(p->darcy->FluxFESpace(), x.GetBlock(0));
   GridFunction t_h(p->darcy->PotentialFESpace(), x.GetBlock(1));

   //heat flux

   std::stringstream ss;
   ss.str("");
   ss << "mesh_" << it << ".mesh";
   std::ofstream ofs(ss.str());
   q_h.FESpace()->GetMesh()->Print(ofs);
   ofs.close();

   q_sock << "solution\n" << *q_h.FESpace()->GetMesh() << q_h << std::endl;
   if (it == 0)
   {
      q_sock << "window_title 'Heat flux'" << std::endl;
      q_sock << "keys Rljvvvvvmmc" << std::endl;
   }

   ss.str("");
   ss << "qh_" << std::setfill('0') << std::setw(5) << it << ".gf";
   q_h.Save(ss.str().c_str());

   //temperature

   t_sock << "solution\n" << *t_h.FESpace()->GetMesh() << t_h << std::endl;
   if (it == 0)
   {
      t_sock << "window_title 'Temperature'" << std::endl;
      t_sock << "keys Rljmmc" << std::endl;
   }

   ss.str("");
   ss << "th_" << std::setfill('0') << std::setw(5) << it << ".gf";
   t_h.Save(ss.str().c_str());
}

}
