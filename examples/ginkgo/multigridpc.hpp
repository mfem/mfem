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
      GINKGO_CUIC = 2,
      GINKGO_CUIC_ISAI = 3
   };
   SolverType type;
#ifdef MFEM_SIMPLEX_LOR
   bool simplex_lor = true;
#endif
   SolverConfig(SolverType type_) : type(type_) { }
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

struct DiffusionMultigrid : Multigrid
{
   Coefficient &coeff;
   OperatorPtr A_coarse;

   DiffusionMultigrid(
      FiniteElementSpaceHierarchy& hierarchy,
      Coefficient &coeff_,
      Array<int>& ess_bdr,
      SolverConfig coarse_solver_config)
       : Multigrid(hierarchy), coeff(coeff_)
    {
       ConstructCoarseOperatorAndSolver(
          coarse_solver_config, hierarchy.GetFESpaceAtLevel(0), ess_bdr);
       int nlevels = hierarchy.GetNumLevels();
       for (int i=1; i<nlevels; ++i)
       {
          ConstructOperatorAndSmoother(hierarchy.GetFESpaceAtLevel(i), ess_bdr);
       }
    }

   void ConstructBilinearForm(
      FiniteElementSpace &fespace, Array<int> &ess_bdr, AssemblyLevel asm_lvl)
    {
       BilinearForm* form = new BilinearForm(&fespace);
       form->SetAssemblyLevel(asm_lvl);
       form->AddDomainIntegrator(new DiffusionIntegrator(coeff));
       form->Assemble();
       bfs.Append(form);
    
       essentialTrueDofs.Append(new Array<int>());
       fespace.GetEssentialTrueDofs(ess_bdr, *essentialTrueDofs.Last());
    }

   void ConstructOperatorAndSmoother(
      FiniteElementSpace& fespace, Array<int>& ess_bdr)
    {
       ConstructBilinearForm(fespace, ess_bdr, AssemblyLevel::PARTIAL);
    
       OperatorPtr opr;
       bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
       opr.SetOperatorOwner(false);
    
       Vector diag(fespace.GetTrueVSize());
       bfs.Last()->AssembleDiagonal(diag);
    
       Solver* smoother = new OperatorChebyshevSmoother(
          opr.Ptr(), diag, *essentialTrueDofs.Last(), 2);
    
       AddLevel(opr.Ptr(), smoother, true, true);
    }

   void ConstructCoarseOperatorAndSolver(
      SolverConfig config, FiniteElementSpace& fespace, Array<int>& ess_bdr)
    {
       ConstructBilinearForm(fespace, ess_bdr, AssemblyLevel::LEGACYFULL);
       BilinearForm &a = *bfs.Last();
       Array<int> &ess_dofs = *essentialTrueDofs.Last();
    
       bfs.Last()->FormSystemMatrix(*essentialTrueDofs.Last(), A_coarse);
    
       OperatorPtr A_prec;
//       if (NeedsLOR(config))
///       {
//          Mesh &mesh = *fespace.GetMesh();
//          int order = fespace.GetOrder(0); // <-- Assume uniform p
//          lor.reset(new LOR(mesh, order, coeff, ess_dofs, config.simplex_lor));
//          A_prec = lor->A;
//       }
//       else
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
          case SolverConfig::FA_AMGX:
          case SolverConfig::LOR_AMGX:
          {
             AmgXSolver *amg = new AmgXSolver;
             amg->ReadParameters("amgx.json", AmgXSolver::EXTERNAL);
             amg->InitExclusiveGPU(MPI_COMM_WORLD);
             amg->SetOperator(*A_prec.As<HypreParMatrix>());
             coarse_solver = amg;
             break;
          }
    #endif
          default:
             MFEM_ABORT("Not available.")
       }
    
       AddLevel(A_coarse.Ptr(), coarse_solver, false, true);
    }
};
}
#endif                                                             
