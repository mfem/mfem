#include "linear_solver.hpp"

namespace mfem
{
void EllipticSolver::BuildEssTdofList()
{

}
EllipticSolver::EllipticSolver(BilinearForm &a, Array<int> &ess_bdr):a(a)
{
   a.FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   SetupSolver();
}

EllipticSolver::EllipticSolver(BilinearForm &a, Array2D<int> &ess_bdr):a(a),
   ess_tdof_list(0)
{
   MFEM_ASSERT(ess_bdr.NumRows() != a.FESpace()->GetVDim()+1,
               "Boundary data should have 1+vdim size (all, 1st, ..., last)");
   Array<int> ess_tdof_list_comp;
   Array<int> ess_bdr_comp;
   for (int i=0; i<ess_bdr.NumRows() + 1; i++)
   {
      ess_bdr.GetRow(i, ess_bdr_comp);
      a.FESpace()->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i-1);
      ess_tdof_list.Append(ess_tdof_list_comp);
   }
   SetupSolver();
}

void EllipticSolver::SetupSolver()
{
   a.FormSystemMatrix(ess_tdof_list, A);

   parallel = false;
#ifdef MFEM_USE_MPI
   par_a = dynamic_cast<ParBilinearForm*>(&a);
   if (par_a)
   {
      parallel = true;
      comm = par_a->ParFESpace()->GetComm();

      par_solver.reset(new HyprePCG(comm));
      par_prec.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));

      par_solver->SetTol(1e-12);
      par_solver->SetMaxIter(2000);
      par_solver->SetPrintLevel(0);
      par_solver->SetOperator(*A.As<HypreParMatrix>());
      par_solver->SetPreconditioner(*par_prec);
      par_prec->SetPrintLevel(0);
      par_B.SetSize(par_a->ParFESpace()->GetTrueVSize());
   }
   else
   {
      parallel = false;

      solver.reset(new CGSolver());
      prec.reset(new GSSmoother(*A.As<SparseMatrix>()));

      solver->SetAbsTol(1e-12);
      solver->SetRelTol(1e-8);
      solver->SetMaxIter(2000);
      solver->SetPrintLevel(0);
      solver->SetOperator(*A.As<SparseMatrix>());
      solver->SetPreconditioner(*prec);
      solver->iterative_mode=true;
   }
#else
   solver.reset(new CGSolver());
   prec.reset(new GSSmoother(*A.As<SparseMatrix>()));

   solver->SetAbsTol(1e-12);
   solver->SetRelTol(1e-8);
   solver->SetMaxIter(2000);
   solver->SetPrintLevel(0);
   solver->SetOperator(*A.As<SparseMatrix>());
   solver->SetPreconditioner(*prec);
#endif
}

void EllipticSolver::UseElasticityOption()
{
#ifdef MFEM_USE_MPI
   if (!parallel) { MFEM_ABORT("Tried to use elasticity option for HypreBoomerAMG, but operator is serial bilinear form."); }
   par_prec->SetElasticityOptions(par_a->ParFESpace());
#else
   MFEM_ABORT("Tried to use elasticity option for HypreBoomerAMG, but MFEM is serial version.");
#endif
}

void EllipticSolver::Solve(LinearForm &b, GridFunction &x)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      static_cast<ParLinearForm*>(&b)->ParallelAssemble(par_B);
      X.MakeRef(static_cast<ParGridFunction*>(&x)->GetTrueVector(), 0, par_B.Size());
      par_a->EliminateVDofsInRHS(ess_tdof_list, X, par_B);
      par_solver->Mult(par_B, X);
      par_a->RecoverFEMSolution(X, b, x);
   }
   else
   {
      B.MakeRef(b, 0, b.Size());
      X.MakeRef(x, 0, x.Size());
      a.EliminateVDofsInRHS(ess_tdof_list, X, B);
      solver->Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
   }
#else
   B.MakeRef(b, 0, b.Size());
   X.MakeRef(x, 0, x.Size());
   a.EliminateVDofsInRHS(ess_tdof_list, X, B);
   solver->Mult(B, X);
   a.RecoverFEMSolution(X, b, x);
#endif
}

}
