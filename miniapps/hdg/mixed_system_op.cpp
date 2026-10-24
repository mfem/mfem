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

#include "mixed_system_op.hpp"
#include "../../general/tic_toc.hpp"
#include <fstream>

#define USE_DIRECT_SOLVER_HYBRIDIZATION
#define USE_DIRECT_SOLVER_REDUCTION
#define USE_DIRECT_SOLVER_SCHUR

namespace mfem
{
namespace hdg
{

void MixedSystemOperator::SetupNonlinearSolver(real_t rtol_, real_t atol_,
                                               int max_iters_)
{
#ifdef MFEM_USE_MPI
   MPI_Comm comm = (pmixed_system)?(pmixed_system->ParFluxFESpace()->GetComm()):
                   (MPI_COMM_NULL);
#endif
   IterativeSolver *lin_solver = NULL;
   switch (solver_type)
   {
      case SolverType::Default:
      case SolverType::LBFGS:
         prec = NULL;
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            solver.reset(new LBFGSSolver(comm));
         }
         else
#endif
            solver.reset(new LBFGSSolver());
         solver_str = "LBFGS";
         break;
      case SolverType::LBB:
         prec = NULL;
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            solver.reset(new LBBSolver(comm));
         }
         else
#endif
            solver.reset(new LBBSolver());
         solver_str = "LBB";
         break;
      case SolverType::Newton:
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            lin_solver = new GMRESSolver(comm);
         }
         else
#endif
            lin_solver = new GMRESSolver();
         prec_str = "GMRES";
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            solver.reset(new NewtonSolver(comm));
         }
         else
#endif
            solver.reset(new NewtonSolver());
         solver_str = "Newton";
         break;
      case SolverType::KINSol:
#ifdef MFEM_USE_SUNDIALS
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            lin_solver = new GMRESSolver(comm);
         }
         else
#endif
            lin_solver = new GMRESSolver();
         prec_str = "GMRES";
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            solver.reset(new KINSolver(comm, KIN_PICARD));
         }
         else
#endif
            solver.reset(new KINSolver(KIN_PICARD));
         static_cast<KINSolver*>(solver.get())->EnableAndersonAcc(10);
         solver_str = "KINSol";
#else
         MFEM_ABORT("Sundials not installed!");
#endif
         break;
   }

   if (lin_solver)
   {
      if (mixed_system->GetHybridization())
      {
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            auto *amg = new HypreBoomerAMG();
            amg->SetAdvectiveOptions();
            lin_prec.reset(amg);
            lin_prec_str = "HypreAMG";
         }
         else
#endif
         {
            lin_prec.reset(new GSSmoother());
            lin_prec_str = "GS";
         }
      }
      else
      {
         SchurPreconditioner *schur;
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            schur = new SchurPreconditioner(pmixed_system, true);
         }
         else
#endif
            schur = new SchurPreconditioner(mixed_system, true);
         lin_prec.reset(schur);
         lin_prec_str = schur->GetString();
         lin_solver->SetPreconditioner(*lin_prec);
      }

      lin_solver->SetAbsTol(atol_);
      lin_solver->SetRelTol(rtol_ * 1e-2);
      lin_solver->SetMaxIter(max_iters_);
      lin_solver->SetPrintLevel(0);
      prec.reset(lin_solver);
   }

   solver->SetAbsTol(atol_);
   solver->SetRelTol(rtol_);
   solver->SetMaxIter(max_iters_);
   if (prec) { solver->SetPreconditioner(*prec); }
   solver->SetPrintLevel((btime_u || btime_p)?0:1);
   solver->iterative_mode = true;
}

void MixedSystemOperator::SetupLinearSolver(real_t rtol_, real_t atol_,
                                            int max_iters_)
{
   if (mixed_system->GetHybridization())
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         auto *amg = new HypreBoomerAMG();
         amg->SetAdvectiveOptions();
         prec.reset(amg);
         prec_str = "HypreAMG";
      }
      else
#endif
      {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_HYBRIDIZATION)
         prec.reset(new GSSmoother());
         prec_str = "GS";
#else
         prec.reset(new UMFPackSolver());
         prec_str = "UMFPack";
#endif
      }
   }
   else if (mixed_system->GetReduction())
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         auto *amg = new HypreBoomerAMG();
         amg->SetAdvectiveOptions();
         prec.reset(amg);
         prec_str = "HypreAMG";
      }
      else
#endif
      {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_REDUCTION)
         prec.reset(new GSSmoother());
         prec_str = "GS";
#else
         prec.reset(new UMFPackSolver());
         prec_str = "UMFPack";
#endif
      }
   }
   else
   {
      SchurPreconditioner *schur;
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         schur = new SchurPreconditioner(pmixed_system);
      }
      else
#endif
         schur = new SchurPreconditioner(mixed_system);
      prec.reset(schur);
      prec_str = schur->GetString();
   }

#ifdef MFEM_USE_MPI
   if (pmixed_system)
   {
      MPI_Comm comm = pmixed_system->ParFluxFESpace()->GetComm();
      solver.reset(new GMRESSolver(comm));
   }
   else
#endif
      solver.reset(new GMRESSolver());
   solver_str = "GMRES";
   solver->SetAbsTol(atol_);
   solver->SetRelTol((sol_type == SolutionController::Type::Native)?(rtol_):(0.));
   solver->SetMaxIter(max_iters_);
   if (prec) { solver->SetPreconditioner(*prec); }
   solver->SetPrintLevel((btime_u || btime_p)?0:1);
   solver->iterative_mode = true;

   if (!monitor)
   {
      if (sol_type != SolutionController::Type::Native)
      {
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            monitor.reset(new ParSolutionController(*pmixed_system, x, rhs, sol_type,
                                                    rtol));
         }
         else
#endif
            monitor.reset(new SolutionController(*mixed_system, x, rhs, sol_type, rtol));
      }
      else if (monitor_step >= 0)
      {
#ifdef MFEM_USE_MPI
         if (pmixed_system)
         {
            monitor.reset(new ParIterativeGLVis(*pmixed_system, x, rhs, monitor_step));
         }
         else
#endif
            monitor.reset(new IterativeGLVis(*mixed_system, x, rhs, monitor_step));
      }

      if (monitor)
      {
         solver->SetMonitor(*monitor);
      }
   }
}

MixedSystemOperator::MixedSystemOperator(const Array<int> &ess_flux_tdofs_list_,
                                         MixedSystemForm *mixed_system_,
                                         std::vector<LinearForm*> rhs,
                                         std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs_,
                                         SolverType stype_, bool btime_u_, bool btime_p_)
   : TimeDependentOperator(0, 0., IMPLICIT),
     ess_flux_tdofs_list(ess_flux_tdofs_list_), mixed_system(mixed_system_),
     coeffs(coeffs_), solver_type(stype_), btime_u(btime_u_), btime_p(btime_p_)
{
   offsets = ConstructOffsets(*mixed_system);
   width = height = offsets.Last();

   if (rhs.size() >= 1 && rhs[0])
   {
      g = rhs[0];
   }
   else
   {
      g = mixed_system->GetFluxRHS();
   }

   if (rhs.size() >= 2 && rhs[1])
   {
      f = rhs[1];
   }
   else
   {
      f = mixed_system->GetPotentialRHS();
   }

   if (mixed_system->GetHybridization())
   {
      trace_space = mixed_system->GetHybridization()->ConstraintFESpace();
      if (rhs.size() >= 3) { h = rhs[2]; }
   }

   if (btime_u || btime_p)
      idtcoeff.reset(new FunctionCoefficient([&](const Vector &) { return idt; }));

   if (btime_u)
   {
      BilinearForm *Mq = const_cast<BilinearForm*>(
                            (const_cast<const MixedSystemForm*>(mixed_system))->GetFluxMassForm());
      NonlinearForm *Mqnl = const_cast<NonlinearForm*>(
                               (const_cast<const MixedSystemForm*>(mixed_system))->GetFluxMassNonlinearForm());
      const int dim = mixed_system->FluxFESpace()->GetMesh()->Dimension();
      const int vdim = mixed_system->FluxFESpace()->GetVDim();
      const bool dg = (mixed_system->FluxFESpace()->FEColl()->GetRangeType(
                          dim) == FiniteElement::SCALAR);

      if (dg)
      {
         auto *bfi = new VectorMassIntegrator(*idtcoeff);
         bfi->SetVDim(vdim);
         if (Mq) { Mq->AddDomainIntegrator(bfi); }
         else if (Mqnl) { Mqnl->AddDomainIntegrator(bfi); }
      }
      else
      {
         MFEM_VERIFY(vdim == 1, "Unsupported case");
         auto *bfi = new VectorFEMassIntegrator(*idtcoeff);
         if (Mq) { Mq->AddDomainIntegrator(bfi); }
         else if (Mqnl) { Mqnl->AddDomainIntegrator(bfi); }
      }

      if (Mqnl && trace_space)
      {
         //hybridization must be reconstructed, since the non-linear
         //potential mass must be passed to it
         const int c_vdim = trace_space->GetVDim();
         mixed_system->EnableHybridization(trace_space,
                                           new VectorBlockDiagonalIntegrator(
                                              c_vdim, new NormalTraceJumpIntegrator()),
                                           ess_flux_tdofs_list);
      }

      Mq0.reset(new BilinearForm(mixed_system->FluxFESpace()));
      if (dg)
      {
         auto *bfi = new VectorMassIntegrator(*idtcoeff);
         bfi->SetVDim(vdim);
         Mq0->AddDomainIntegrator(bfi);
      }
      else
      {
         MFEM_VERIFY(vdim == 1, "Unsupported case");
         Mq0->AddDomainIntegrator(new VectorFEMassIntegrator(*idtcoeff));
      }
   }

   if (btime_p)
   {
      const int vdim = mixed_system->PotentialFESpace()->GetVDim();

      BilinearForm *Mt = const_cast<BilinearForm*>(
                            (const_cast<const MixedSystemForm*>(mixed_system))->GetPotentialMassForm());
      NonlinearForm *Mtnl = const_cast<NonlinearForm*>(
                               (const_cast<const MixedSystemForm*>
                                (mixed_system))->GetPotentialMassNonlinearForm());

      auto *bfi = new VectorMassIntegrator(*idtcoeff);
      bfi->SetVDim(vdim);
      if (Mt) { Mt->AddDomainIntegrator(bfi); }
      else if (Mtnl) { Mtnl->AddDomainIntegrator(bfi); }

      if (Mtnl && trace_space)
      {
         //hybridization must be reconstructed, since the non-linear
         //potential mass must be passed to it
         const int c_vdim = trace_space->GetVDim();
         mixed_system->EnableHybridization(trace_space,
                                           new VectorBlockDiagonalIntegrator(
                                              c_vdim, new NormalTraceJumpIntegrator()),
                                           ess_flux_tdofs_list);
      }

      Mt0.reset(new BilinearForm(mixed_system->PotentialFESpace()));
      auto *bfi0 = new VectorMassIntegrator(*idtcoeff);
      bfi0->SetVDim(vdim);
      Mt0->AddDomainIntegrator(bfi0);
   }
}

#ifdef MFEM_USE_MPI
MixedSystemOperator::MixedSystemOperator(const Array<int> &ess_flux_tdofs_list,
                                         ParMixedSystemForm *mixed_system_,
                                         std::vector<ParLinearForm*> rhs,
                                         std::vector<std::variant<Coefficient*,VectorCoefficient*>> coeffs,
                                         SolverType stype, bool bflux_u, bool btime_p)
   : MixedSystemOperator(ess_flux_tdofs_list, (MixedSystemForm*) mixed_system_,
                         std::vector<LinearForm*>(rhs.begin(), rhs.end()), coeffs,
                         stype, bflux_u, btime_p)
{
   pmixed_system = mixed_system_;

   pg = static_cast<ParLinearForm*>(g);
   pf = static_cast<ParLinearForm*>(f);
   ph = static_cast<ParLinearForm*>(h);
}
#endif //MFEM_USE_MPI

MixedSystemOperator::~MixedSystemOperator()
{
}

Array<int> MixedSystemOperator::ConstructOffsets(const MixedSystemForm
                                                 &mixed_system)
{
   if (!mixed_system.GetHybridization())
   {
      return mixed_system.GetOffsets();
   }

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = mixed_system.FluxFESpace()->GetVSize();
   offsets[2] = mixed_system.PotentialFESpace()->GetVSize();
   offsets[3] = mixed_system.GetHybridization()->ConstraintFESpace()->GetVSize();
   offsets.PartialSum();

   return offsets;
}

void MixedSystemOperator::ImplicitSolve(const real_t dt, const Vector &x_v,
                                        Vector &dx_v)
{
#ifdef MFEM_USE_MPI
   const bool verbose = (pmixed_system)?(Mpi::Root()):(true);
#else
   const bool verbose = true;
#endif

   //form the linear system

   rhs.Update(g->GetData(), mixed_system->GetOffsets());
   x.Update(dx_v, mixed_system->GetOffsets());
   dx_v = x_v;

   //set time

   for (const auto &coeff : coeffs)
   {
      if (coeff.index() == 0)
      {
         std::get<0>(coeff)->SetTime(t);
      }
      else if (coeff.index() == 1)
      {
         std::get<1>(coeff)->SetTime(t);
      }
   }

   //assemble rhs

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

#ifdef MFEM_USE_MPI
   if (pmixed_system)
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

      mixed_system->Update();

      //assemble the system
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         pmixed_system->Assemble();
      }
      else
#endif //MFEM_USE_MPI
      {
         mixed_system->Assemble();
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
      u_h.MakeRef(mixed_system->FluxFESpace(), x.GetBlock(0), 0);
      Mq0->AddMult(u_h, *g, +1.);
   }

   if (Mt0)
   {
      GridFunction p_h;
      p_h.MakeRef(mixed_system->PotentialFESpace(), x.GetBlock(1), 0);
      Mt0->AddMult(p_h, *f, -1.);
   }
#if 0
   if (Mq0 && Mt0)
   {
      GridFunction u_h, p_h;
      u_h.MakeRef(mixed_system->FluxFESpace(), x.GetBlock(0), 0);
      p_h.MakeRef(mixed_system->PotentialFESpace(), x.GetBlock(1), 0);
      mixed_system->GetFluxDivForm()->AddMultTranspose(p_h, *g, -1.);
      mixed_system->GetFluxDivForm()->AddMult(u_h, *f, +1.);
   }
#endif
   //form the reduced system

   OperatorHandle op;
   Vector X, RHS;
   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         if (ph)
         {
            RHS.SetSize(trace_space->GetTrueVSize());
            ph->ParallelAssemble(RHS);
         }
      }
      else
#endif
      {
         if (trace_space->GetMesh()->Nonconforming())
         {
            auto *cP = trace_space->GetConformingProlongation();
            RHS.SetSize(cP->Width());
            cP->MultTranspose(*h, RHS);

            auto *cR = trace_space->GetConformingRestriction();
            X.SetSize(cR->Height());
            const Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
            cR->Mult(x_r, X);
         }
         else
         {
            X.MakeRef(dx_v, offsets[2], trace_space->GetVSize());
            if (h) { RHS.MakeRef(*h, 0, trace_space->GetVSize()); }
         }
      }
   }

   mixed_system->FormLinearSystem(ess_flux_tdofs_list, x, rhs,
                                  op, X, RHS, true);


   chrono.Stop();
   if (verbose) { std::cout << "Assembly took " << chrono.RealTime() << "s.\n"; }

   if (reassemble)
   {
      // 10. Construct the preconditioner and solver

      chrono.Clear();
      chrono.Start();

      // We do not want to initialize any new forms here, only obtain
      // the existing ones, so we const cast the MixedSystemForm
      const MixedSystemForm *cmixed_system =
         const_cast<const MixedSystemForm*>(mixed_system);

      //const BilinearForm *Mq = cmixed_system->GetFluxMassForm();
      const NonlinearForm *Mqnl = cmixed_system->GetFluxMassNonlinearForm();
      const BlockNonlinearForm *Mnl = cmixed_system->GetBlockNonlinearForm();
      //const MixedBilinearForm *B = cmixed_system->GetFluxDivForm();
      //const BilinearForm *Mt = cmixed_system->GetPotentialMassForm();
      const NonlinearForm *Mtnl = cmixed_system->GetPotentialMassNonlinearForm();

      if (trace_space) //hybridization
      {
         if (Mqnl || Mtnl || Mnl)
         {
            mixed_system->GetHybridization()->SetLocalNLSolver(
               MixedSystemHybridization::LSsolveType::Newton,
               max_iters, rtol * 1e-3, atol, -1);
            lsolver_str = "Newton+GMRES";

            SetupNonlinearSolver(rtol, atol, max_iters);
         }
         else
         {
            SetupLinearSolver(rtol, atol, max_iters);
         }

         solver->SetOperator(*op);
      }
      else if (mixed_system->GetReduction()) //reduction
      {
         SetupLinearSolver(rtol, atol, max_iters);

         solver->SetOperator(*op);
      }
      else //mixed
      {
         if ((Mqnl || Mtnl || Mnl) && solver_type != SolverType::Default)
         {
            if (prec && ess_flux_tdofs_list.Size() > 0)
            {
               MFEM_ABORT("Gradient is not implemented with essential DOFs!");
            }

            SetupNonlinearSolver(rtol, atol, max_iters);
         }
         else
         {
            if (Mqnl || Mtnl || Mnl)
            {
               std::cerr << "A linear solver is used for a non-linear problem!" << std::endl;
            }

            SetupLinearSolver(rtol, atol, max_iters);
         }
         solver->SetOperator(*op);
      }

      chrono.Stop();
      if (verbose) { std::cout << "Preconditioner took " << chrono.RealTime() << "s.\n"; }
   }

   // 11. Solve the linear system with GMRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();

   solver->Mult(RHS, X);

   mixed_system->RecoverFEMSolution(X, rhs, x);

   if (trace_space)
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
         trace_space->GetProlongationMatrix()->Mult(X, x_r);
      }
      else
#endif
         if (trace_space->GetMesh()->Nonconforming())
         {
            Vector x_r(dx_v, offsets[2], trace_space->GetVSize());
            auto *cP = trace_space->GetConformingProlongation();
            cP->Mult(X, x_r);
         }
   }

   chrono.Stop();

   if (verbose)
   {
      std::cout << solver_str;
      if (!prec_str.empty()) { std::cout << "+" << prec_str; }
      if (!lin_prec_str.empty()) { std::cout << "+" << lin_prec_str; }
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
      std::cout << "Solver took " << chrono.RealTime() << "s.\n";
   }

   dx_v -= x_v;
   dx_v *= idt;
}

void MixedSystemOperator::Update()
{
   offsets = ConstructOffsets(*mixed_system);
   width = height = offsets.Last();

   if (Mt0) { Mt0->Update(); }
   if (Mq0) { Mq0->Update(); }

   idt = 0.;
}

MixedSystemOperator::SchurPreconditioner::SchurPreconditioner(
   const MixedSystemForm *mixed_system_, bool nonlinear_)
   : Solver(mixed_system_->Height()), mixed_system(mixed_system_),
     nonlinear(nonlinear_)
{
   if (!nonlinear)
   {
      Vector x_(Width());
      x_ = 0.;
      Construct(x_);
   }

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   if (pa)
   {
      prec_str = "OperJacobi";
   }
   else
   {
#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_SCHUR)
      prec_str = "GS";
#else
      prec_str = "UMFPack";
#endif
   }
}

#ifdef MFEM_USE_MPI
MixedSystemOperator::SchurPreconditioner::SchurPreconditioner(
   const ParMixedSystemForm *mixed_system_, bool nonlinear_)
   : Solver(mixed_system_->Height()), mixed_system(mixed_system_),
     pmixed_system(mixed_system_), nonlinear(nonlinear_)
{
   if (!nonlinear)
   {
      Vector x_(Width());
      x_ = 0.;
      ConstructPar(x_);
   }

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   if (pa)
   {
      prec_str = "OperJacobi";
   }
   else
   {
      prec_str = "HypreAMG";
   }
}
#endif //MFEM_USE_MPI

void MixedSystemOperator::SchurPreconditioner::Mult(const Vector &x_,
                                                    Vector &y_) const
{
   if (nonlinear && reconstruct)
   {
#ifdef MFEM_USE_MPI
      if (pmixed_system)
      {
         ConstructPar(x_);
      }
      else
#endif //MFEM_USE_MPI
      {
         Construct(x_);
      }
      reconstruct = false;
   }
   schur_prec->Mult(x_,y_);
}

void MixedSystemOperator::SchurPreconditioner::Construct(
   const Vector &x_v) const
{
   const Array<int> &block_offsets = mixed_system->GetTrueOffsets();
   BlockVector bx(x_v.GetData(), block_offsets);

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const BilinearForm *Mq = mixed_system->GetFluxMassForm();
   const NonlinearForm *Mqnl = mixed_system->GetFluxMassNonlinearForm();
   const BlockNonlinearForm *Mnl = mixed_system->GetBlockNonlinearForm();
   const MixedBilinearForm *B = mixed_system->GetFluxDivForm();
   const BilinearForm *Mt = mixed_system->GetPotentialMassForm();
   const NonlinearForm *Mtnl = mixed_system->GetPotentialMassNonlinearForm();

   const int a_tsize = block_offsets[1] - block_offsets[0];
   const int d_tsize = block_offsets[2] - block_offsets[1];
   Vector Md(a_tsize);
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Md.Size());
      for (int i=0; i < invMd.Size(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(d_tsize);
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
   }
   else
   {
      const BlockOperator *bop = NULL;
      const BlockOperator *bgrad = NULL;

      if (nonlinear)
      {
         bop = dynamic_cast<const BlockOperator*>(op);
         if (!bop)
         {
            auto *grad = dynamic_cast<const MixedSystemForm::Gradient*>(op);
            MFEM_ASSERT(grad, "Unknown gradient operator!");
            bop = &grad->BlockMatrices();
         }
      }

      // get diagonal
      const SparseMatrix *Mqm;
      if (bop)
      {
         Mqm = static_cast<const SparseMatrix*>(&bop->GetBlock(0,0));
      }
      else if (Mq)
      {
         Mqm = &Mq->SpMat();
      }
      else if (Mqnl)
      {
         Mqm = static_cast<SparseMatrix*>(&Mqnl->GetGradient(bx.GetBlock(0)));
      }
      else if (Mnl)
      {
         bgrad = static_cast<const BlockOperator*>(&Mnl->GetGradient(bx));
         Mqm = static_cast<const SparseMatrix*>(&bgrad->GetBlock(0,0));
      }
      else
      {
         MFEM_ABORT("No flux diagonal!");
      }

      Mqm->GetDiag(Md);
      invM = new DSmoother(*Mqm);

      Md.HostReadWrite();

      const SparseMatrix &Bm(B->SpMat());
      SparseMatrix *MinvBt = Transpose(Bm);

      for (int i = 0; i < Md.Size(); i++)
      {
         MinvBt->ScaleRow(i, 1./Md(i));
      }

      S.reset(mfem::Mult(Bm, *MinvBt));
      delete MinvBt;

      const SparseMatrix *Mtm = NULL;
      if (bop && !bop->IsZeroBlock(1,1))
      {
         Mtm = static_cast<const SparseMatrix*>(&bop->GetBlock(1,1));
      }
      else if (Mt)
      {
         Mtm = &Mt->SpMat();
      }
      else if (Mtnl)
      {
         Mtm = static_cast<SparseMatrix*>(&Mtnl->GetGradient(bx.GetBlock(1)));
      }

      if (Mtm)
      {
         S.reset(Add(*Mtm, *S));
      }

      if (!bop && Mnl)
      {
         Mtm = static_cast<const SparseMatrix*>(&bgrad->GetBlock(1,1));
         if (Mtm->NumNonZeroElems() > 0)
         {
            S.reset(Add(*Mtm, *S));
         }
      }

#if !defined(MFEM_USE_SUITESPARSE) or !defined(USE_DIRECT_SOLVER_SCHUR)
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   schur_prec.reset(new BlockDiagonalPreconditioner(block_offsets));
   schur_prec->owns_blocks = true;
   schur_prec->SetDiagonalBlock(0, invM);
   schur_prec->SetDiagonalBlock(1, invS);
}

#ifdef MFEM_USE_MPI
void MixedSystemOperator::SchurPreconditioner::ConstructPar(
   const Vector &x_v) const
{
   const Array<int> &block_offsets = pmixed_system->GetTrueOffsets();
   BlockVector bx(x_v.GetData(), block_offsets);

   // Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     temperature Schur Complement

   const bool pa = (mixed_system->GetAssemblyLevel() != AssemblyLevel::LEGACY);

   const ParBilinearForm *Mq = pmixed_system->GetParFluxMassForm();
   const ParNonlinearForm *Mqnl = pmixed_system->GetParFluxMassNonlinearForm();
   const ParBlockNonlinearForm *Mnl = pmixed_system->GetParBlockNonlinearForm();
   const ParMixedBilinearForm *B = pmixed_system->GetParFluxDivForm();
   const ParBilinearForm *Mt = pmixed_system->GetParPotentialMassForm();
   const ParNonlinearForm *Mtnl =
      pmixed_system->GetParPotentialMassNonlinearForm();

   const int a_tsize = block_offsets[1] - block_offsets[0];
   const int d_tsize = block_offsets[2] - block_offsets[1];
   Vector Md(a_tsize);
   schur_prec.reset(new BlockDiagonalPreconditioner(block_offsets));
   schur_prec->owns_blocks = true;
   Solver *invM, *invS;

   if (pa)
   {
      Mq->AssembleDiagonal(Md);
      auto Md_host = Md.HostRead();
      Vector invMd(Md.Size());
      for (int i=0; i < invMd.Size(); ++i)
      {
         invMd(i) = 1.0 / Md_host[i];
      }

      Vector BMBt_diag(d_tsize);
      B->AssembleDiagonal_ADAt(invMd, BMBt_diag);

      Array<int> ess_tdof_list;  // empty

      invM = new OperatorJacobiSmoother(Md, ess_tdof_list);
      invS = new OperatorJacobiSmoother(BMBt_diag, ess_tdof_list);
   }
   else
   {
      const BlockOperator *bop = NULL;
      const BlockOperator *bgrad = NULL;

      if (nonlinear)
      {
         bop = dynamic_cast<const BlockOperator*>(op);
         if (!bop)
         {
            auto *grad = dynamic_cast<const ParMixedSystemForm::ParGradient*>(op);
            MFEM_ASSERT(grad, "Unknown gradient operator!");
            bop = &grad->BlockMatrices();
         }
      }

      // get diagonal
      const HypreParMatrix *Mqm;
      if (bop)
      {
         Mqm = static_cast<const HypreParMatrix*>(&bop->GetBlock(0,0));
      }
      else if (Mq)
      {
         Mqm = const_cast<ParBilinearForm*>(Mq)->ParallelAssembleInternalMatrix();
      }
      else if (Mqnl)
      {
         Mqm = static_cast<const HypreParMatrix*>(&Mqnl->GetGradient(bx.GetBlock(0)));
      }
      else if (Mnl)
      {
         bgrad = &Mnl->GetGradient(bx);
         Mqm = static_cast<const HypreParMatrix*>(&bgrad->GetBlock(0,0));
      }
      else
      {
         MFEM_ABORT("No flux diagonal!");
      }

      Mqm->GetDiag(Md);
      invM = new HypreDiagScale(*Mqm);

      Md.HostReadWrite();

      const HypreParMatrix *Bm;
      if (bop)
      {
         Bm = static_cast<const HypreParMatrix*>(&bop->GetBlock(1,0));
      }
      else if (B)
      {
         Bm = const_cast<ParMixedBilinearForm*>(B)->ParallelAssembleInternalMatrix();
      }
      else
      {
         MFEM_ABORT("No flux divergence!");
      }
      HypreParMatrix *MinvBt = Bm->Transpose();
      MinvBt->InvScaleRows(Md);

      hS.reset(mfem::ParMult(Bm, MinvBt, true));
      delete MinvBt;

      const HypreParMatrix *Mtm = NULL;
      if (bop && !bop->IsZeroBlock(1,1))
      {
         Mtm = static_cast<const HypreParMatrix*>(&bop->GetBlock(1,1));
      }
      else if (Mt)
      {
         Mtm = const_cast<ParBilinearForm*>(Mt)->ParallelAssembleInternalMatrix();
      }
      else if (Mtnl)
      {
         Mtm = static_cast<HypreParMatrix*>(&Mtnl->GetGradient(bx.GetBlock(1)));
      }

      if (Mtm)
      {
         hS.reset(ParAdd(Mtm, hS.get()));
      }

      if (!bop && Mnl)
      {
         Mtm = static_cast<const HypreParMatrix*>(&bgrad->GetBlock(1,1));
         if (Mtm && Mtm->NNZ() > 0)
         {
            hS.reset(ParAdd(Mtm, hS.get()));
         }
      }

      {
         auto *amg = new HypreBoomerAMG(*hS);
         amg->SetAdvectiveOptions();
         amg->SetPrintLevel(0);
         invS = amg;
      }
   }

   invM->iterative_mode = false;
   invS->iterative_mode = false;

   schur_prec->SetDiagonalBlock(0, invM);
   schur_prec->SetDiagonalBlock(1, invS);
}
#endif

MixedSystemOperator::SolutionController::SolutionController(
   MixedSystemForm &mixed_system_, BlockVector &x_, const BlockVector &rhs_,
   Type type_,
   real_t rtol_)
   : mixed_system(mixed_system_), x(x_), rhs(rhs_), type(type_), rtol(rtol_)
{
   switch (type)
   {
      case Type::Native:
         break;
      case Type::Flux:
         sol_prev.SetSize(mixed_system.FluxFESpace()->GetTrueVSize());
         break;
      case Type::Potential:
         sol_prev.SetSize(mixed_system.PotentialFESpace()->GetTrueVSize());
         break;
   }
}

bool MixedSystemOperator::SolutionController::CheckSolution(const Vector &x,
                                                            const Vector &y) const
{
   real_t vals[2];
   real_t &diff = vals[0], &sum = vals[1];

   for (int i = 0; i < x.Size(); i++)
   {
      const real_t dx = x(i) - y(i);
      const real_t avg = (x(i) + y(i)) / 2.;
      diff += dx*dx;
      sum += avg*avg;
   }

   this->ReduceValues(vals, 2);

   return diff < sum * (rtol*rtol);
}

void MixedSystemOperator::SolutionController::MonitorSolution(
   int it, real_t norm, const Vector &X, bool final)
{
   if (type == Type::Native || converged) { return; }

   mixed_system.RecoverFEMSolution(X, rhs, x);

   Vector &sol = x.GetBlock((type == Type::Flux)?(0):(1));

   if (it > it_prev)
   {
      converged = CheckSolution(sol, sol_prev);
   }

   it_prev = it;
   sol_prev = sol;
}

#ifdef MFEM_USE_MPI
MixedSystemOperator::ParSolutionController::ParSolutionController(
   ParMixedSystemForm &pmixed_system_, BlockVector &x_, const BlockVector &rhs_,
   Type type_,
   real_t rtol_)
   : SolutionController(pmixed_system_, x_, rhs_, type_, rtol_),
     pmixed_system(pmixed_system_) { }

void MixedSystemOperator::ParSolutionController::ReduceValues(real_t vals[],
                                                              int num) const
{
   MPI_Comm comm = pmixed_system.ParFluxFESpace()->GetComm();
   MPI_Allreduce(MPI_IN_PLACE, vals, 2, MFEM_MPI_REAL_T, MPI_SUM, comm);
}

void MixedSystemOperator::ParSolutionController::MonitorSolution(
   int it, real_t norm, const Vector &X, bool final)
{
   if (type == Type::Native || converged) { return; }

   mixed_system.RecoverFEMSolution(X, rhs, x);

   Vector &sol_x = x.GetBlock((type == Type::Flux)?(0):(1));
   ParFiniteElementSpace *fes = (type == Type::Flux)?
                                (pmixed_system.ParFluxFESpace()):
                                (pmixed_system.ParPotentialFESpace());
   Vector sol(fes->GetTrueVSize());

   fes->GetRestrictionOperator()->Mult(sol_x, sol);

   if (it > it_prev)
   {
      converged = CheckSolution(sol, sol_prev);
   }

   it_prev = it;
   sol_prev = sol;
}
#endif //MFEM_USE_MPI

MixedSystemOperator::IterativeGLVis::IterativeGLVis(
   MixedSystemForm &mixed_system_, BlockVector &x_, const BlockVector &rhs_,
   int step_, bool save_files_)
   : mixed_system(mixed_system_), x(x_), rhs(&rhs_), step(step_),
     save_files(save_files_)
{
   Init();
}

MixedSystemOperator::IterativeGLVis::IterativeGLVis(
   MixedSystemForm &mixed_system_, BlockVector &x_, int step_, bool save_files_)
   : mixed_system(mixed_system_), x(x_), rhs(NULL), step(step_),
     save_files(save_files_)
{
   Init();
}

void MixedSystemOperator::IterativeGLVis::Init()
{
   const char vishost[] = "localhost";
   const int  visport   = 19916;
   q_sock.open(vishost, visport);
   q_sock.precision(8);
   t_sock.open(vishost, visport);
   t_sock.precision(8);
}

std::string MixedSystemOperator::IterativeGLVis::FormFilename(const char *base,
                                                              int it, const char *suff)
{
   std::stringstream ss;
   ss << base << "_" << std::setfill('0') << std::setw(5) << it << "." << suff;
   return ss.str();
}

void MixedSystemOperator::IterativeGLVis::MonitorSolution(int it, real_t norm,
                                                          const Vector &X, bool final)
{
   if (step != 0 && it % step != 0 && !final) { return; }

   if (rhs)
   {
      mixed_system.RecoverFEMSolution(X, *rhs, x);
   }
   else
   {
      mixed_system.RecoverFEMSolution(X, x);
   }

   GridFunction q_h(mixed_system.FluxFESpace(), x.GetBlock(0));
   GridFunction t_h(mixed_system.PotentialFESpace(), x.GetBlock(1));

   //heat flux

   if (save_files)
   {
      std::ofstream ofs(this->FormFilename("mesh", it, "mesh"));
      q_h.FESpace()->GetMesh()->Print(ofs);
      ofs.close();
   }

   this->StreamPreamble(q_sock);
   q_sock << "solution\n" << *q_h.FESpace()->GetMesh() << q_h << std::endl;
   if (it == 0)
   {
      q_sock << "window_title 'Heat flux'" << std::endl;
      q_sock << "keys Rljvvvvvmmc" << std::endl;
   }

   if (save_files)
   {
      q_h.Save(this->FormFilename("qh", it).c_str());
   }

   //temperature

   this->StreamPreamble(t_sock);
   t_sock << "solution\n" << *t_h.FESpace()->GetMesh() << t_h << std::endl;
   if (it == 0)
   {
      t_sock << "window_title 'Temperature'" << std::endl;
      t_sock << "keys Rljmmc" << std::endl;
   }

   if (save_files)
   {
      t_h.Save(this->FormFilename("th", it).c_str());
   }
}

#ifdef MFEM_USE_MPI
void MixedSystemOperator::ParIterativeGLVis::StreamPreamble(socketstream &ss)
{
   const int num_procs = pmixed_system.ParFluxFESpace()->GetNRanks();
   const int myid = pmixed_system.ParFluxFESpace()->GetMyRank();
   ss << "parallel " << num_procs << " " << myid << "\n";
}

std::string MixedSystemOperator::ParIterativeGLVis::FormFilename(
   const char *base, int it, const char *suff)
{
   std::stringstream ss;
   const int myid = pmixed_system.ParFluxFESpace()->GetMyRank();
   ss << base << "_" << std::setfill('0') << std::setw(6) << myid
      << "_" << std::setw(5) << it << "." << suff;
   return ss.str();
}
#endif // MFEM_USE_MPI

void RandomizeMesh(Mesh &mesh, real_t dr)
{
   const FiniteElementSpace *fes_v = mesh.GetNodalFESpace();
   if (fes_v)
   {
      MFEM_VERIFY(fes_v, "Not a nodal mesh");
      MFEM_VERIFY(fes_v->FEColl()->GetContType() ==
                  FiniteElementCollection::CONTINUOUS, "Only continuous mappings are supported");
      MFEM_VERIFY(fes_v->GetOrdering() == Ordering::byVDIM,
                  "Only ordering by VDIM is supported");
      MFEM_VERIFY(fes_v->GetVDim() == 2, "Only 2D is supported at the moment");
   }

   std::default_random_engine eng(0); //<------
   std::uniform_real_distribution<real_t> dist(-1., 1.);

   // vertices
   const int NV = mesh.GetNV();
   const Table *e2v = mesh.GetEdgeVertexTable();
   const int *I = e2v->GetI();
   const int *J = e2v->GetJ();
   GridFunction *nodes = mesh.GetNodes();
   const int vdim = mesh.SpaceDimension();
   real_t dx_max = 0.;
   for (int v = 0; v < NV; v++)
   {
      Array<int> verts;
      verts.Reserve(4);
      for (int i = 0; i < e2v->Size(); i++)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            if (J[j] == v) { verts.Append(i); }
         }
      }
      if (verts.Size() < 4) { continue; } //<--------

      Vector x_min(vdim), x_max(vdim);
      Vector x_v(mesh.GetVertex(v), vdim);
      x_min = x_v;
      x_max = x_v;
      for (int i : verts)
      {
         for (int j = I[i]; j < I[i+1]; j++)
         {
            if (J[j] == v) { continue; }
            const real_t *vert = mesh.GetVertex(J[j]);
            for (int d = 0; d < vdim; d++)
            {
               if (vert[d] < x_min[d]) { x_min[d] = vert[d]; }
               if (vert[d] > x_max[d]) { x_max[d] = vert[d]; }
            }
         }
      }
      for (int d = 0; d < vdim; d++)
      {
         const real_t r = dist(eng);
         const real_t dx = (r < 0.)?(x_v[d] - x_min[d]):(x_max[d] - x_v[d]);
         if (dx > dx_max) { dx_max = dx; }
         const real_t dv = r * dr * dx;
         if (nodes)
         {
            (*nodes)(v*vdim + d) += dv;
         }
         x_v(d) += dv;
      }
   }

   if (fes_v)
   {
      // interior nodes
      const int NE = mesh.GetNEdges();
      Array<int> dofs;
      for (int e = 0; e < NE; e++)
      {
         int el1, el2;
         mesh.GetFaceElements(e, &el1, &el2);
         if (el2 < 0) { continue; }
         fes_v->GetEdgeDofs(e, dofs);
         Vector xc(vdim);
         for (int i = 0; i < dofs.Size() - 2; i++)
         {
            // to node
            const real_t dri = dist(eng) * dr;
            int node;
            if (dri > 0.)
            {
               node = (i+1 >= dofs.Size())?(1):(i+1);
            }
            else
            {
               node = (i-1 <= 1)?(0):(i-1);
            }

            for (int d = 0; d < vdim; d++)
            {
               real_t &x_i = (*nodes)(dofs[2+i]*vdim + d);
               const real_t &x_n = (*nodes)(dofs[node]*vdim + d);
               x_i += fabs(dri) * (x_n - x_i);
            }

            // to center
            const real_t drc = dist(eng) * dr;
            mesh.GetElementCenter((drc > 0.)?(el2):(el1), xc);
            for (int d = 0; d < vdim; d++)
            {
               real_t &x_i = (*nodes)(dofs[2+i]*vdim + d);
               x_i += fabs(drc) * (xc(d) - x_i);
            }
         }
      }

      mesh.NodesUpdated();
   }
}

} // namespace hdg
} // namespace mfem
