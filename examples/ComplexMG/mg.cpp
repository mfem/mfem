#include "mfem.hpp"
#include "mg.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


MGSolver::MGSolver(HypreParMatrix * Af_, std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_->Height(), Af_->Width()), Af(Af_), P(P_) {

   StopWatch chrono;
   NumGrids = P.size();
   S.resize(NumGrids);
   A.resize(NumGrids + 1);

   A[NumGrids] = Af;
   for (int i = NumGrids ; i > 0; i--)
   {
      A[i - 1] = RAP(A[i], P[i - 1]);
   }

   mumps = new MUMPSSolver;
   mumps->SetPrintLevel(0);
   mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps->SetOperator(*A[0]);
   invAc = mumps;


   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      // S[i] = new SchwarzSmoother(fespaces[i]->GetParMesh(),1,fespaces[i+1],A[i+1]);
      S[i] = new SchwarzSmoother(fespaces[i+1]->GetParMesh(),0,fespaces[i+1],A[i+1]);
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

MGSolver::~MGSolver() 
{
   int n = S.size();
   for (int i = n - 1; i >= 0 ; i--)
   {
      delete S[i];
      delete A[i];
   }
   S.clear();
   A.clear();
   delete invAc;
}

ComplexMGSolver::ComplexMGSolver(ComplexHypreParMatrix * Af_, 
std::vector<HypreParMatrix *> P_,std::vector<ParFiniteElementSpace * > fespaces)
   : Solver(Af_->Height(), Af_->Width()), Af(Af_), P(P_) {

   NumGrids = P.size();
   S.resize(NumGrids);
   A.resize(NumGrids + 1);

   A[NumGrids] = Af;
   for (int i = NumGrids ; i > 0; i--)
   {
      // A[i - 1] = RAP(A[i], P[i - 1]);
      A[i - 1] = new ComplexHypreParMatrix(RAP(&A[i]->real(), P[i - 1]), 
                                           RAP(&A[i]->imag(), P[i - 1]), true, true);
   }

   mumps = new ComplexMUMPSSolver;
   mumps->SetPrintLevel(0);
   mumps->SetOperator(*A[0]);
   invAc = mumps;


   for (int i = NumGrids - 1; i >= 0 ; i--)
   {
      S[i] = new ComplexSchwarzSmoother(fespaces[i]->GetParMesh(),1,fespaces[i+1],A[i+1]);
      // S[i] = new ComplexSchwarzSmoother(fespaces[i+1]->GetParMesh(),0,fespaces[i+1],A[i+1]);
   }

}

void ComplexMGSolver::Mult(const Vector &r, Vector &z) const
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
   Vector rv0, rv1;
   for (int i = NumGrids; i > 0 ; i--)
   {
      // Pre smooth
      S[i - 1]->Mult(rv[i], zv[i]); zv[i] *= theta;
      // compute residual
      Vector w(A[i]->Height());
      A[i]->Mult(zv[i], w);
      rv[i] -= w;
      // Restrict

      double * data1 = rv[i].GetData();
      double * data0 = rv[i-1].GetData();
      int size1 = rv[i].Size();
      int size0 = rv[i-1].Size();

      // Real part
      rv1.SetDataAndSize(data1, size1/2);
      rv0.SetDataAndSize(data0, size0/2);
      P[i - 1]->MultTranspose(rv1, rv0);

      // Imag part
      rv1.SetDataAndSize(&data1[size1/2], size1/2);
      rv0.SetDataAndSize(&data0[size0/2], size0/2);
      P[i - 1]->MultTranspose(rv1, rv0);
      // P[i - 1]->MultTranspose(rv[i], rv[i - 1]);
   }

   // Coarse grid Solve
   invAc->Mult(rv[0], zv[0]);
   //
   Vector u0,u1;
   for (int i = 1; i <= NumGrids ; i++)
   {
      // Prolong correction
      Vector u(2*P[i - 1]->Height());
      // real part
      double * data1 = u.GetData();
      double * data0 = zv[i - 1].GetData();
      int size1 = u.Size();
      int size0 = zv[i-1].Size();

      u1.SetDataAndSize(data1, size1/2);
      u0.SetDataAndSize(data0, size0/2);

      P[i - 1]->Mult(u0, u1);

      // imag part

      u1.SetDataAndSize(&data1[size1/2], size1/2);
      u0.SetDataAndSize(&data0[size0/2], size0/2);

      P[i - 1]->Mult(u0, u1);

      // P[i - 1]->Mult(zv[i - 1], u);
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

ComplexMGSolver::~ComplexMGSolver() 
{
   int n = S.size();
   for (int i = n - 1; i >= 0 ; i--)
   {
      delete S[i];
      delete A[i];
   }
   S.clear();
   A.clear();
   delete invAc;
}