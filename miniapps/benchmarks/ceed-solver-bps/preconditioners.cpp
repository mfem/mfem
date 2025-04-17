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

#include "preconditioners.hpp"

namespace mfem
{

AssemblyLevel GetCoarseAssemblyLevel(SolverConfig config)
{
   switch (config.type)
   {
      case SolverConfig::JACOBI:
      case SolverConfig::LOR_HYPRE:
      case SolverConfig::LOR_AMGX:
         return AssemblyLevel::PARTIAL;
      default:
         return AssemblyLevel::FULL;
         // return AssemblyLevel::LEGACYFULL;
   }
}

bool NeedsLOR(SolverConfig config)
{
   switch (config.type)
   {
      case SolverConfig::LOR_HYPRE:
      case SolverConfig::LOR_AMGX:
         return true;
      default:
         return false;
   }
}

DiffusionMultigrid::DiffusionMultigrid(
   ParFiniteElementSpaceHierarchy& hierarchy,
   Coefficient &coeff_,
   Array<int>& ess_bdr,
   SolverConfig coarse_solver_config,
   int q1d_inc_,
   int smoothers_cheby_order_)
   : GeometricMultigrid(hierarchy, ess_bdr),
     coeff(coeff_),
     q1d_inc(q1d_inc_),
     irs(0, Quadrature1D::GaussLegendre),
     smoothers_cheby_order(smoothers_cheby_order_)
{
   ConstructCoarseOperatorAndSolver(
      coarse_solver_config, hierarchy.GetFESpaceAtLevel(0), ess_bdr);
   int nlevels = hierarchy.GetNumLevels();
   for (int i=1; i<nlevels; ++i)
   {
      ConstructOperatorAndSmoother(hierarchy.GetFESpaceAtLevel(i), ess_bdr);
   }
}

void DiffusionMultigrid::ConstructBilinearForm(
   ParFiniteElementSpace &fespace, Array<int> &ess_bdr, AssemblyLevel asm_lvl)
{
   ParBilinearForm *form = new ParBilinearForm(&fespace);
   form->SetAssemblyLevel(asm_lvl);

   DiffusionIntegrator *integ = new DiffusionIntegrator(coeff);

   int p = fespace.GetOrder(0);
   int dim = fespace.GetMesh()->Dimension();
   // Integration rule for high-order problem: (p+1+q1d_inc)^d Gauss-Legendre
   // points
   int int_order = 2*(p+1+q1d_inc) - 1;
   Geometry::Type geom = fespace.GetMesh()->GetElementBaseGeometry(0);
   const IntegrationRule &ir = irs.Get(geom, int_order);
   MFEM_VERIFY(ir.Size() == pow(p+1+q1d_inc,dim), "Wrong quadrature");
   integ->SetIntegrationRule(ir);

   form->AddDomainIntegrator(integ);
   form->Assemble();
   bfs.Append(form);

   essentialTrueDofs.Append(new Array<int>());
   fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
}

void DiffusionMultigrid::ConstructOperatorAndSmoother(
   ParFiniteElementSpace& fespace, Array<int>& ess_bdr)
{
   ConstructBilinearForm(fespace, ess_bdr, AssemblyLevel::PARTIAL);

   OperatorPtr opr;
   bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
   opr.SetOperatorOwner(false);

   Vector diag(fespace.GetTrueVSize());
   bfs.Last()->AssembleDiagonal(diag);

   Solver* smoother = new OperatorChebyshevSmoother(
      *opr, diag, *essentialTrueDofs.Last(), smoothers_cheby_order,
      fespace.GetParMesh()->GetComm());

   AddLevel(opr.Ptr(), smoother, true, true);
}

void DiffusionMultigrid::ConstructCoarseOperatorAndSolver(
   SolverConfig config, ParFiniteElementSpace& fespace, Array<int>& ess_bdr)
{
   ConstructBilinearForm(fespace, ess_bdr, GetCoarseAssemblyLevel(config));
   ParBilinearForm &a = static_cast<ParBilinearForm&>(*bfs.Last());
   Array<int> &ess_dofs = *essentialTrueDofs.Last();

   a.FormSystemMatrix(ess_dofs, A_coarse);

   OperatorPtr A_prec;
   if (NeedsLOR(config))
   {
      if (Mpi::Root())
      {
         std::cout << "Forming LOR discretization..." << std::endl;
      }
      lor.reset(new ParLORDiscretization(a, ess_dofs));
      A_prec = lor->GetAssembledSystem();
      if (Mpi::Root())
      {
         std::cout << "Forming LOR discretization... Done." << std::endl;
      }
   }
   else
   {
      A_prec = A_coarse;
   }

   if (Mpi::Root()) { std::cout << "Forming preconditioner... " << std::endl; }
   switch (config.type)
   {
      case SolverConfig::JACOBI:
         coarse_precond.reset(new OperatorJacobiSmoother(a, ess_dofs));
         break;
      case SolverConfig::FA_HYPRE:
      case SolverConfig::LOR_HYPRE:
      {
         HypreBoomerAMG *amg = new HypreBoomerAMG(*A_prec.As<HypreParMatrix>());
         amg->SetPrintLevel(1);
         Vector b(amg->Height());
         Vector x(amg->Height());
         b = 0.0;
         x = 0.0;
         amg->Setup(b, x); // Force setup;
         coarse_precond.reset(amg);
         break;
      }
#ifdef MFEM_USE_AMGX
      case SolverConfig::FA_AMGX:
      case SolverConfig::LOR_AMGX:
      {
         AmgXSolver *amg = new AmgXSolver;
         amg->ReadParameters(config.amgx_config_file, AmgXSolver::EXTERNAL);
         amg->InitExclusiveGPU(MPI_COMM_WORLD);
         amg->SetOperator(*A_prec.As<HypreParMatrix>());
         coarse_precond.reset(amg);
         break;
      }
#endif
      default:
         MFEM_ABORT("Not available.")
   }

   if (config.inner_sli) // coarse_solver = SLI
   {
      SLISolver *sli = new SLISolver(fespace.GetComm());
      sli->SetPrintLevel(0);
      sli->SetAbsTol(0.0);
      sli->SetRelTol(0.0);
      sli->SetMaxIter(config.inner_sli_iter);
      sli->SetOperator(*A_coarse);
      sli->SetPreconditioner(*coarse_precond);
      coarse_solver.reset(sli);
   }
   else if (config.inner_cg)
   {
      CGSolver *cg = new CGSolver(MPI_COMM_WORLD);
      cg->SetPrintLevel(2);
      cg->SetMaxIter(100);
      cg->SetRelTol(1e-8);
      cg->SetAbsTol(0.0);
      cg->SetOperator(*A_coarse);
      cg->SetPreconditioner(*coarse_precond);
      cg->iterative_mode = false;
      coarse_solver.reset(cg);
   }
   else
   {
      coarse_solver = coarse_precond;
   }
   if (Mpi::Root())
   {
      std::cout << "Forming preconditioner... Done.\n" << std::endl;
   }

   if (config.coarse_smooth)
   {
      Vector diag(fespace.GetTrueVSize());
      a.AssembleDiagonal(diag);

      Solver *smoother = new OperatorChebyshevSmoother(
         *A_coarse, diag, ess_dofs, smoothers_cheby_order,
         fespace.GetParMesh()->GetComm());

      AddLevel(A_coarse.Ptr(), smoother, false, true);
      AddCoarseSolver(coarse_solver.get(), false);
   }
   else
   {
      AddLevel(A_coarse.Ptr(), coarse_solver.get(), false, false);
   }
}

void DiffusionMultigrid::SetSmoothersChebyshevOrder(int new_cheby_order)
{
   for (int level = MultigridBase::coarse_solver ? 0 : 1;
        level < NumLevels(); level++)
   {
      OperatorChebyshevSmoother *cheby =
         dynamic_cast<OperatorChebyshevSmoother*>(GetSmootherAtLevel(level));
      if (cheby) { cheby->SetOrder(new_cheby_order); }
   }
   smoothers_cheby_order = new_cheby_order;
}

void DiffusionMultigrid::SetInnerSLINumIter(int inner_sli_iter)
{
   SLISolver *sli = dynamic_cast<SLISolver*>(coarse_solver.get());
   if (sli) { sli->SetMaxIter(inner_sli_iter); }
}

} // namespace mfem
