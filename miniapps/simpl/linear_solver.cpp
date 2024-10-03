#include "linear_solver.hpp"

namespace mfem
{
void EllipticSolver::BuildEssTdofList()
{

}
EllipticSolver::EllipticSolver(BilinearForm &a, Array<int> &ess_bdr):a(a)
{
   a.FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   a.FormSystemMatrix(ess_tdof_list, A);
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
   a.FormSystemMatrix(ess_tdof_list, A);
   parallel = false;
#ifdef MFEM_USE_MPI
   par_a = dynamic_cast<ParBilinearForm*>(&a);
   if (par_a)
   {
      parallel = true;
   }
#endif
}

void EllipticSolver::SetupSolver()
{
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

void EllipticSolver::Solve(LinearForm &b, GridFunction &x)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      a.EliminateVDofsInRHS(ess_tdof_list, x, b);
      par_solver->Mult(x, b);
   }
#endif
}
Solver* EllipticSolver::GetSolver()
{
   return nullptr;
}
}
