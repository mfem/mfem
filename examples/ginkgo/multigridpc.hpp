#ifndef __MULTIGRIDPC_HPP__
#define __MULTIGRIDPC_HPP__

#include "mfem.hpp"
#include <memory>

namespace mfem
{

struct SolverConfig
{
   enum SolverType
   {
      JACOBI = 0,
      AMGX = 1,
      CHEBYSHEV = 2,
      GINKGO_CUIC = 3,
      GINKGO_CUIC_ISAI = 4
   };
   SolverType type;
   SolverType smoother_type;
   AssemblyLevel upper_level_asm;
   const char *amgx_config;
   Ginkgo::GinkgoExecutor gko_exec;
   SolverConfig(SolverType type_, SolverType sm_type_, AssemblyLevel upper_asm_,
                const char *amgx_config_, Device &device) : type(type_),
      smoother_type(sm_type_),
      upper_level_asm(upper_asm_),
      gko_exec(device)
   {
      amgx_config = amgx_config_;
   }
};

struct MGRefinement
{
   enum Type { P_MG, H_MG };
   Type type;
   int order;
   MGRefinement(Type type_, int order_) : type(type_), order(order_) { }
   static MGRefinement p(int order_) { return MGRefinement(P_MG, order_); }
   static MGRefinement h() { return MGRefinement(H_MG, 0); }
};

bool NeedsLOR(SolverConfig config)
{
   switch (config.type)
   {
      case SolverConfig::GINKGO_CUIC:
      case SolverConfig::GINKGO_CUIC_ISAI:
         return true;
      default:
         return false;
   }
}

struct DiffusionMultigrid : GeometricMultigrid
{
   Coefficient &coeff;
   OperatorPtr A_coarse;

   DiffusionMultigrid(
      FiniteElementSpaceHierarchy& hierarchy,
      Coefficient &coeff_,
      Array<int>& ess_bdr,
      SolverConfig solver_config)
      : GeometricMultigrid(hierarchy), coeff(coeff_)
   {
      ConstructCoarseOperatorAndSolver(
         solver_config, hierarchy.GetFESpaceAtLevel(0), ess_bdr);
      int nlevels = hierarchy.GetNumLevels();
      for (int i=1; i<nlevels; ++i)
      {
         ConstructOperatorAndSmoother(solver_config, hierarchy.GetFESpaceAtLevel(i),
                                      ess_bdr);
      }
   }

   void ConstructBilinearForm(
      FiniteElementSpace &fespace, Array<int> &ess_bdr, AssemblyLevel asm_lvl)
   {
      BilinearForm* form = new BilinearForm(&fespace);
      form->SetAssemblyLevel(asm_lvl);
      form->SetDiagonalPolicy(DIAG_ONE);
      form->AddDomainIntegrator(new DiffusionIntegrator(coeff));
      form->Assemble();
      bfs.Append(form);

      essentialTrueDofs.Append(new Array<int>());
      fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
   }

   void ConstructOperatorAndSmoother(SolverConfig solver_config,
                                     FiniteElementSpace& fespace, Array<int>& ess_bdr)
   {
      ConstructBilinearForm(fespace, ess_bdr, solver_config.upper_level_asm);

      OperatorPtr opr;
      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      switch (solver_config.smoother_type)
      {
         case SolverConfig::CHEBYSHEV:
         {
            Vector diag(fespace.GetTrueVSize());
            bfs.Last()->AssembleDiagonal(diag);

            Solver* smoother = new OperatorChebyshevSmoother(
               opr.Ptr(), diag, *essentialTrueDofs.Last(), 2);

            if (solver_config.upper_level_asm == AssemblyLevel::PARTIAL)
            {
               AddLevel(opr.Ptr(), smoother, true, true);
            }
            else
            {
               AddLevel(opr.Ptr(), smoother, false, true);
            }
            break;
         }
         case SolverConfig::GINKGO_CUIC:
         {
            SparseMatrix *A_lvl = dynamic_cast<SparseMatrix*>(opr.Ptr());
            Solver *smoother = new Ginkgo::IcPreconditioner(
               solver_config.gko_exec, "exact");
            smoother->SetOperator(*A_lvl);
            if (solver_config.upper_level_asm == AssemblyLevel::PARTIAL)
            {
               AddLevel(opr.Ptr(), smoother, true, true);
            }
            else
            {
               AddLevel(opr.Ptr(), smoother, false, true);
            }
            break;
         }
         case SolverConfig::GINKGO_CUIC_ISAI:
         {
            SparseMatrix *A_lvl = dynamic_cast<SparseMatrix*>(opr.Ptr());
            Solver *smoother = new Ginkgo::IcIsaiPreconditioner(
               solver_config.gko_exec, "exact");
            smoother->SetOperator(*A_lvl);
            if (solver_config.upper_level_asm == AssemblyLevel::PARTIAL)
            {
               AddLevel(opr.Ptr(), smoother, true, true);
            }
            else
            {
               AddLevel(opr.Ptr(), smoother, false, true);
            }
            break;
         }
      }

   }

   void ConstructCoarseOperatorAndSolver(
      SolverConfig config, FiniteElementSpace& fespace, Array<int>& ess_bdr)
   {
      ConstructBilinearForm(fespace, ess_bdr, AssemblyLevel::LEGACYFULL);
      BilinearForm &a = *bfs.Last();
      Array<int> &ess_dofs = *essentialTrueDofs.Last();

      bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), A_coarse);

      OperatorPtr A_prec;
      {
         A_prec = A_coarse;
      }

      Solver *coarse_solver;
      switch (config.type)
      {
         case SolverConfig::JACOBI:
            coarse_solver = new OperatorJacobiSmoother(a, ess_dofs);
            break;
#ifdef MFEM_USE_AMGX
         case SolverConfig::AMGX:
         {
            AmgXSolver *amg = new AmgXSolver;
            amg->ReadParameters(config.amgx_config, AmgXSolver::EXTERNAL);
            amg->InitSerial();
            amg->SetOperator(*A_prec.As<SparseMatrix>());
            coarse_solver = amg;
            break;
         }
#endif
         case SolverConfig::GINKGO_CUIC:
         {
            SparseMatrix *A_lvl = dynamic_cast<SparseMatrix*>(A_prec.Ptr());
            Solver *gko_solver = new Ginkgo::IcPreconditioner(
               config.gko_exec, "exact");
            gko_solver->SetOperator(*A_lvl);
            coarse_solver = gko_solver;
            break;
         }
         case SolverConfig::GINKGO_CUIC_ISAI:
         {
            SparseMatrix *A_lvl = dynamic_cast<SparseMatrix*>(A_prec.Ptr());
            Solver *gko_solver = new Ginkgo::IcIsaiPreconditioner(
               config.gko_exec, "exact");
            gko_solver->SetOperator(*A_lvl);
            coarse_solver = gko_solver;
            break;
         }
         default:
            MFEM_ABORT("Not available.")
      }

      AddLevel(A_coarse.Ptr(), coarse_solver, false, true);
   }
};
}
#endif
