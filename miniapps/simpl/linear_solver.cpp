#include "linear_solver.hpp"

namespace mfem
{
EllipticSolver::EllipticSolver(BilinearForm &a, Array<int> &ess_tdof_list):a(a),
   ess_tdof_list(ess_tdof_list)
{
   a.FormSystemMatrix(ess_tdof_list, A);

   parallel = false;
#ifdef MFEM_USE_MPI
   par_a = dynamic_cast<ParBilinearForm*>(&a);
   if (par_a)
   {
      parallel = true;
      comm = par_a->ParFESpace()->GetComm();
   }
#endif

   if (parallel)
   {
#ifdef MFEM_USE_MPI
      par_solver.reset(new CGSolver(comm));
      par_prec.reset(new HypreBoomerAMG(*A.As<HypreParMatrix>()));

      par_solver->SetAbsTol(1e-12);
      par_solver->SetRelTol(1e-6);
      par_solver->SetMaxIter(2000);
      par_solver->SetPrintLevel(0);
      par_solver->SetOperator(*A.As<HypreParMatrix>());
      par_solver->SetPreconditioner(*par_prec);
      par_solver->iterative_mode = true;
      if (a.FESpace()->GetVDim() > 1)
      {
         par_prec->SetSystemsOptions(a.FESpace()->GetVDim(),
                                     a.FESpace()->GetOrdering() == Ordering::byNODES);
      }
      par_prec->SetPrintLevel(0);
      par_B.SetSize(par_a->ParFESpace()->GetTrueVSize());
#endif
   }
   else
   {
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
}

void EllipticProblem::BuildTDofList(Array<int> &ess_bdr)
{
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
}

void EllipticProblem::BuildTDofList(Array2D<int> &ess_bdr)
{
   ess_tdof_list.SetSize(0);
   MFEM_ASSERT(ess_bdr.NumRows() == fes.GetVDim()+1,
               "Boundary data should have 1+vdim size (all, 1st, ..., last)");
   Array<int> ess_tdof_list_comp;
   Array<int> ess_bdr_comp;
   for (int i=0; i<ess_bdr.NumRows(); i++)
   {
      ess_bdr.GetRow(i, ess_bdr_comp);
      fes.GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i-1);
      ess_tdof_list.Append(ess_tdof_list_comp);
   }
}

void EllipticProblem::InitializeForms(int nRHS)
{
   b.resize(nRHS);
   if (hasAdjoint) { adjb.resize(nRHS); }

#ifdef MFEM_USE_MPI
   par_fes = dynamic_cast<ParFiniteElementSpace*>(&fes);
   if (par_fes)
   {
      parallel = true;
      comm = par_fes->GetComm();
      par_a = new ParBilinearForm(par_fes);
      a.reset(par_a);
      for (int i=0; i<nRHS; i++)
      {
         par_b.push_back(new ParLinearForm(par_fes));
         b[i].reset(par_b[i]);
         if (hasAdjoint)
         {
            par_adjb.push_back(new ParLinearForm(par_fes));
            adjb[i].reset(par_adjb[i]);
         }
      }
   }
   else
   {
      a.reset(new BilinearForm(&fes));
      for (int i=0; i<nRHS; i++)
      {
         b[i].reset(new LinearForm(&fes));
         if (hasAdjoint) { adjb[i].reset(new LinearForm(&fes)); }
      }
   }
#else
   a.reset(new BilinearForm(&fes));
   b.reset(new LinearForm(&fes));
   if (hasAdjoint)
   {
      adjb.reset(new LinearForm(&fes));
   }
#endif
}

void EllipticProblem::Solve(GridFunction &x, bool reuse_solver, int i)
{
   if (!reuse_solver)
   {
      if (!isAStationary) {a->Update(); a->Assemble();}
      if (!solver || !isAStationary)
      {
         ResetSolver();
      }
   }
   if (!isBStationary) {b[i]->Assemble();}
   solver->Solve(*b[i], x);
}

void EllipticProblem::SolveAdjoint(GridFunction &x, bool reuse_solver, int i)
{
   MFEM_ASSERT(hasAdjoint,
               "SolveAdjoint(GridFunction &) is called without setting hasAdjoint=true.");
   if (!reuse_solver)
   {
      if (!isAStationary) {a->Update(); a->Assemble();}
      if (!solver || !isAStationary)
      {
         ResetSolver();
      }
   }
   if (!isAdjBStationary) {adjb[i]->Assemble();}
   solver->Solve(*adjb[i], x);
}

void EllipticSolver::UseElasticityOption()
{
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      par_prec->SetElasticityOptions(par_a->ParFESpace());
#endif
   }
   else
   {
      MFEM_ABORT("Tried to use elasticity option for HypreBoomerAMG, but parallel option is turned off");
   }
}

void EllipticSolver::Solve(LinearForm &b, GridFunction &x)
{
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      static_cast<ParLinearForm*>(&b)->ParallelAssemble(par_B);
      X.MakeRef(static_cast<ParGridFunction*>(&x)->GetTrueVector(), 0, par_B.Size());
      par_a->EliminateVDofsInRHS(ess_tdof_list, X, par_B);
      par_solver->Mult(par_B, X);
      par_a->RecoverFEMSolution(X, b, x);
#endif
   }
   else
   {
      B.MakeRef(b, 0, b.Size());
      X.MakeRef(x, 0, x.Size());
      a.EliminateVDofsInRHS(ess_tdof_list, X, B);
      solver->Mult(B, X);
      a.RecoverFEMSolution(X, b, x);
   }
}

void EllipticSolver::Update()
{
   a.FormSystemMatrix(ess_tdof_list, A);

   if (parallel)
   {
#ifdef MFEM_USE_MPI
      par_solver->SetOperator(*A.As<HypreParMatrix>());
      par_prec->SetOperator(*A.As<HypreParMatrix>());
#endif
   }
   else
   {
      solver->SetOperator(*A.As<SparseMatrix>());
      prec->SetOperator(*A.As<SparseMatrix>());
   }
}

}
