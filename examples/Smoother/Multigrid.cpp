#include "mfem.hpp"
#include "Multigrid.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


MGSolver::MGSolver(HypreParMatrix * Af_, std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_->Height(), Af_->Width()), Af(Af_), P(P_) {

   NumGrids = P.size();
   S.resize(NumGrids);
   A.resize(NumGrids + 1);

   A[NumGrids] = Af;
   for (int i = NumGrids ; i > 0; i--)
   {
      A[i - 1] = RAP(A[i], P[i - 1]);
   }
   // Set up coarse solve operator
   petsc = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   // Convert to PetscParMatrix
   petsc->SetOperator(PetscParMatrix(A[0], Operator::PETSC_MATAIJ));
   invAc = petsc;
   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      S[i] = new ParSchwarzSmoother(fespaces[i+1]->GetParMesh(),0,fespaces[i+1],A[i+1]);
      S[i]->SetDumpingParam(1.0/5.0);
      // S[i]->SetType(HypreSmoother::Jacobi);
      // S[i]->SetOperator(*A[i+1]);
   }
}

void MGSolver::Mult(const Vector &r, Vector &z) const
{
   // Residual vectors
   std::vector<Vector> rv(NumGrids + 1);
   // correction vectors
   std::vector<Vector> zv(NumGrids + 1);
   // allocation
   for (int i = 0; i <= NumGrids ; i++)
   {
      int n = A[i]->Width();
      rv[i].SetSize(n);
      zv[i].SetSize(n);
   }
   // Initial residual
   rv[NumGrids] = r;
   // smooth and update residuals down to the coarsest level
   for (int i = NumGrids; i > 0 ; i--)
   {
      // Pre smooth
      S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
      // compute residual
      Vector w(A[i]->Height());
      A[i]->Mult(zv[i], w);
      rv[i] -= w;
      // Restrict
      P[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   }

   // Coarse grid Solve
   invAc->Mult(rv[0], zv[0]);
   //
   for (int i = 1; i <= NumGrids ; i++)
   {
      // Prolong correction
      Vector u(P[i - 1]->Height());
      P[i - 1]->Mult(zv[i - 1], u);
      // Update correction
      zv[i] += u;
      // Update residual
      Vector v(A[i]->Height());
      A[i]->Mult(u, v); rv[i] -= v;
      // Post smooth
      S[i - 1]->Mult(rv[i], v); v *= theta;
      // Update correction
      zv[i] += v;
   }
   z = zv[NumGrids];
}

MGSolver::~MGSolver() {
   int n = S.size();
   for (int i = n - 1; i >= 0 ; i--)
   {
      delete S[i];
   }
}




BlockMGSolver::BlockMGSolver(Array2D<HypreParMatrix *> Af_, std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_(0,0)->Height()+Af_(1,0)->Height(), Af_(0,0)->Width()+Af_(1,0)->Width()), Af(Af_), P(P_) {

   // NumGrids = P.size();
   // S.resize(NumGrids);
   // A.resize(NumGrids + 1);

   // A[NumGrids] = Af;
   // for (int i = NumGrids ; i > 0; i--)
   // {
   //    A[i - 1] = RAP(A[i], P[i - 1]);
   // }
   // // Set up coarse solve operator
   // petsc = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   // // Convert to PetscParMatrix
   // petsc->SetOperator(PetscParMatrix(A[0], Operator::PETSC_MATAIJ));
   // invAc = petsc;
   // for (int i = NumGrids - 1; i >= 0 ; i--)
   // {
   //    S[i] = new ParSchwarzSmoother(fespaces[i+1]->GetParMesh(),0,fespaces[i+1],A[i+1]);
   //    S[i]->SetDumpingParam(1.0/5.0);
   //    // S[i]->SetType(HypreSmoother::Jacobi);
   //    // S[i]->SetOperator(*A[i+1]);
   // }
}

void BlockMGSolver::Mult(const Vector &r, Vector &z) const
{
   // Residual vectors
   // std::vector<Vector> rv(NumGrids + 1);
   // // correction vectors
   // std::vector<Vector> zv(NumGrids + 1);
   // // allocation
   // for (int i = 0; i <= NumGrids ; i++)
   // {
   //    int n = A[i]->Width();
   //    rv[i].SetSize(n);
   //    zv[i].SetSize(n);
   // }
   // // Initial residual
   // rv[NumGrids] = r;
   // // smooth and update residuals down to the coarsest level
   // for (int i = NumGrids; i > 0 ; i--)
   // {
   //    // Pre smooth
   //    S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
   //    // compute residual
   //    Vector w(A[i]->Height());
   //    A[i]->Mult(zv[i], w);
   //    rv[i] -= w;
   //    // Restrict
   //    P[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   // }

   // // Coarse grid Solve
   // invAc->Mult(rv[0], zv[0]);
   // //
   // for (int i = 1; i <= NumGrids ; i++)
   // {
   //    // Prolong correction
   //    Vector u(P[i - 1]->Height());
   //    P[i - 1]->Mult(zv[i - 1], u);
   //    // Update correction
   //    zv[i] += u;
   //    // Update residual
   //    Vector v(A[i]->Height());
   //    A[i]->Mult(u, v); rv[i] -= v;
   //    // Post smooth
   //    S[i - 1]->Mult(rv[i], v); v *= theta;
   //    // Update correction
   //    zv[i] += v;
   // }
   // z = zv[NumGrids];
}

MGSolver::~MGSolver() {
   int n = S.size();
   for (int i = n - 1; i >= 0 ; i--)
   {
      delete S[i];
   }
}